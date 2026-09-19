"""
ЯЗЫКОВАЯ ГОЛОВА АГЕНТА.

Две задачи, обе решаются поверх того же наблюдения, что и выбор действия:

  1. ПОНЯТЬ команду человека: "сделай каменный меч" -> цель stone_sword.
  2. ОБЪЯСНИТЬ, что агент делает: состояние -> "рублю дерево, 3 из 4".

Важно, чего здесь НЕТ: это не языковая модель общего назначения. Агент не
сочиняет текст — он выбирает из набора реплик и подставляет числа. Зато
он не может соврать: каждая реплика собирается из реальных полей
наблюдения, а не выдумывается. Для цели "галлюцинации около нуля" это
принципиально: выдумывать попросту нечем.

Размер: эмбеддинги словаря + два небольших слоя, порядка 200 тысяч
параметров против 42 миллионов у зрения.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..spaces import DENSE_DIM, ITEMS, SL_COORDS, SL_SURVIVAL
from .vocab import (BOS_ID, EOS_ID, N_VOCAB, PAD_ID, RU_ITEMS, RU_KINDS,
                    RU_QUESTIONS, RU_TIERS, RU_VERBS, encode, lookup,
                    tokenize)

# Набор реплик. Агент выбирает индекс, числа подставляются из наблюдения —
# поэтому фраза всегда правдива.
REPLIES: List[str] = [
    "ищу {target}",
    "рублю дерево, {have} из {need}",
    "копаю {target}",
    "иду к верстаку, {dist} блоков",
    "кладу {item} в слот {slot}",
    "беру {item} из сетки",
    "крафчу {target}",
    "готово, сделал {target}",
    "нужно ещё {need} {target}",
    "не хватает {item}",
    "дерусь с {mob}",
    "убегаю от {mob}",
    "ем, голод {food} из 20",
    "здоровье {hp} из 20, опасно",
    "жду, не вижу что делать",
    "не могу это сделать",
    "цель: {target}",
    "у меня {have} {item}",
    "рядом {mob}",
    "не вижу врагов",
]
N_REPLIES = len(REPLIES)


@dataclass
class LangConfig:
    """Настройки языковой головы."""
    emb: int = 48
    hidden: int = 128
    max_len: int = 24
    n_goals: int = 24


class LanguageHead(nn.Module):
    """
    Понимание команд и выбор реплики.

    Вход — токены команды и текущее наблюдение (dense-вектор).
    Выход — какая цель имелась в виду и что ответить.
    """

    def __init__(self, cfg: LangConfig = LangConfig()):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(N_VOCAB, cfg.emb, padding_idx=PAD_ID)
        # Позиция слова важна: "доска над палкой" и "палка над доской" —
        # разные вещи. Без позиций фраза превратилась бы в мешок слов.
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.emb))

        self.text_enc = nn.Sequential(
            nn.Linear(cfg.emb, cfg.hidden), nn.SiLU(),
        )
        # Состояние мира сжимаем сильно: языковой голове не нужны детали,
        # нужен общий контекст (что в руках, сколько здоровья, где цель).
        self.state_enc = nn.Sequential(
            nn.Linear(DENSE_DIM, cfg.hidden), nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(cfg.hidden * 2, cfg.hidden), nn.SiLU(),
        )
        self.goal_head = nn.Linear(cfg.hidden, cfg.n_goals)
        self.reply_head = nn.Linear(cfg.hidden, N_REPLIES)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.05)

    def forward(self, tokens: torch.Tensor, dense: torch.Tensor):
        """
        tokens — (B, max_len) int64, dense — (B, DENSE_DIM) float32.
        Возвращает (логиты целей, логиты реплик).
        """
        e = self.emb(tokens) + self.pos[:, : tokens.shape[1]]
        # Среднее по непустым токенам: фраза короткая, сложный энкодер
        # только добавил бы параметров без выигрыша.
        mask = (tokens != PAD_ID).float().unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        t = self.text_enc(pooled)
        s = self.state_enc(dense)
        h = self.trunk(torch.cat([t, s], dim=-1))
        return self.goal_head(h), self.reply_head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Разбор команды правилами.
#
# Нейросеть учится понимать команды, но пока она не обучена, правила уже
# работают. Заодно они дают разметку для обучения: команда -> цель.
# --------------------------------------------------------------------------
def parse_command(text: str) -> Dict[str, Optional[str]]:
    """
    Разбирает команду человека в структуру.

    Возвращает {"verb", "target", "tier", "kind", "count", "question"}.
    Ничего не выдумывает: чего в тексте нет, того нет и в ответе.
    """
    from .vocab import _stem
    toks = tokenize(text)
    # Падежные формы: "верстаку" -> "верстак", "брёвен" -> "бревно".
    # Сопоставляем и слово целиком, и его основу с основами словарей.
    def _match(d):
        for t in toks:
            if t in d:
                return d[t]
        st_toks = [_stem(t) for t in toks]
        for key, val in d.items():
            ks = _stem(key)
            for st in st_toks:
                if st and (st == ks or st.startswith(ks) or ks.startswith(st)):
                    if min(len(st), len(ks)) >= 4:
                        return val
        return None
    out: Dict[str, Optional[str]] = {
        "verb": None, "target": None, "tier": None,
        "kind": None, "count": None, "question": None,
    }
    for t in toks:
        if t.isdigit():
            out["count"] = t
    out["question"] = _match(RU_QUESTIONS)
    out["verb"] = _match(RU_VERBS)
    out["tier"] = _match(RU_TIERS)
    out["kind"] = _match(RU_KINDS)
    out["target"] = _match(RU_ITEMS)

    # "каменный меч" -> stone_sword: материал плюс вид инструмента.
    if out["tier"] and out["kind"]:
        combo = f"{out['tier']}_{out['kind']}"
        if combo in ITEMS:
            out["target"] = combo
    return out


def command_to_goal(text: str, goals: List[str]) -> Optional[int]:
    """Номер цели из списка GOALS по команде. None, если не понял."""
    p = parse_command(text)
    tgt = p.get("target")
    if tgt and tgt in goals:
        return goals.index(tgt)
    return None


# Внутренние имена в человеческие. Агент отвечает человеку, а не логу.
_RU_NAME: Dict[str, Tuple[str, str]] = {
    "oak_log": ("бревно", "брёвен"),
    "oak_planks": ("доску", "досок"),
    "stick": ("палку", "палок"),
    "crafting_table": ("верстак", "верстаков"),
    "cobblestone": ("камень", "камней"),
    "raw_iron": ("железо", "железа"),
    "iron_ingot": ("слиток", "слитков"),
    "diamond": ("алмаз", "алмазов"),
    "wooden_sword": ("деревянный меч", "мечей"),
    "stone_sword": ("каменный меч", "мечей"),
    "iron_sword": ("железный меч", "мечей"),
    "diamond_sword": ("алмазный меч", "мечей"),
    "wooden_pickaxe": ("деревянную кирку", "кирок"),
    "stone_pickaxe": ("каменную кирку", "кирок"),
    "iron_pickaxe": ("железную кирку", "кирок"),
}


def _ru(name: str, count: int = 1) -> str:
    """Человеческое имя предмета; при счёте — форма множественного числа."""
    pair = _RU_NAME.get(name)
    if not pair:
        return name.replace("_", " ")
    return pair[1] if count > 1 else pair[0]


def describe(obs: Dict[str, np.ndarray], goal_name: str,
             extra: Optional[Dict] = None) -> str:
    """
    Строит честную фразу о происходящем ПРЯМО ИЗ наблюдения.

    Ключевое свойство: все числа берутся из obs, ничего не выдумывается.
    Если агент говорит "здоровье 7 из 20" — значит, в наблюдении ровно 7.
    """
    extra = extra or {}
    dense = obs["dense"]
    surv = dense[SL_SURVIVAL]
    coords = dense[SL_COORDS]

    hp = round(float(surv[0]) * 20)
    food = round(float(surv[1]) * 20)
    # surv[3] — "есть ли враждебный в поле зрения", surv[4] — насколько
    # близко (1 = вплотную). Бой упоминаем только когда враг ДЕЙСТВИТЕЛЬНО
    # рядом, иначе агент рапортует о драке, стоя один в поле.
    enemy_seen = float(surv[3]) > 0.5
    enemy_close = float(surv[4]) > 0.7
    dist_table = round((1.0 - float(coords[5])) * 17)

    # Порядок важен: сначала то, что угрожает жизни, потом задача.
    if hp <= 6:
        return f"здоровье {hp} из 20, опасно"
    if enemy_seen and enemy_close:
        mob = extra.get("mob", "врагом")
        return f"дерусь с {mob}" if hp > 10 else f"убегаю от {mob}"
    if food <= 6:
        return f"ем, голод {food} из 20"
    if extra.get("crafted"):
        return f"готово, сделал {_ru(goal_name)}"

    have = int(extra.get("have", 0))
    need = int(extra.get("need", 0))
    if need and have < need:
        left = need - have
        return f"нужно ещё {left} {_ru(goal_name, left)}"
    if dist_table > 1 and extra.get("going_to_table"):
        return f"иду к верстаку, {dist_table} блоков"
    if enemy_seen:
        return f"вижу {extra.get('mob', 'врага')}, держусь в стороне"
    return f"цель: {_ru(goal_name)}"
