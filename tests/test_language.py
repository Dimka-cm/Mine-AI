"""Тесты языкового модуля: понимание команд и честность ответов."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.env.mc_env import GOALS, MinecraftCraftEnv  # noqa: E402
from brain.language.head import (LangConfig, LanguageHead,  # noqa: E402
                                 command_to_goal, describe, parse_command)
from brain.language.vocab import (N_VOCAB, decode, encode,  # noqa: E402
                                  lookup, tokenize)
from brain.rewards.db import RewardDB  # noqa: E402
from brain.rewards.engine import RewardEngine  # noqa: E402
from brain.rewards.rules import seed  # noqa: E402


@pytest.fixture
def env(tmp_path):
    db = RewardDB(str(tmp_path / "t.db"))
    seed(db)
    return MinecraftCraftEnv(RewardEngine(db), max_steps=90, seed=7)


# --- словарь --------------------------------------------------------------
def test_vocab_is_small_and_covers_domain():
    """Словарь компактный, но покрывает предметную область."""
    assert 200 < N_VOCAB < 1000, f"словарь {N_VOCAB} токенов"
    for word in ("меч", "кирка", "бревно", "зомби", "верстак"):
        assert lookup(word) != lookup("<unk>"), f"{word} должен быть в словаре"


def test_cases_are_understood():
    """Падежные формы сводятся к той же основе."""
    pairs = [("бревно", "брёвен"), ("верстак", "верстаку"),
             ("доска", "досок"), ("палка", "палки")]
    for base, form in pairs:
        assert lookup(base) == lookup(form), f"{base} != {form}"


def test_yo_and_ye_are_same():
    """ё и е — одна буква, иначе 'брёвен' и 'бревен' разошлись бы."""
    assert tokenize("брёвен") == tokenize("бревен")


# --- разбор команд --------------------------------------------------------
@pytest.mark.parametrize("text,goal", [
    ("сделай каменный меч", "stone_sword"),
    ("скрафти деревянную кирку", "wooden_pickaxe"),
    ("добудь 5 брёвен", "oak_log"),
    ("иди к верстаку", "crafting_table"),
    ("нужны палки", "stick"),
])
def test_command_maps_to_goal(text, goal):
    """Команда человека превращается в конкретную цель."""
    assert command_to_goal(text, GOALS) == GOALS.index(goal)


def test_tier_and_kind_combine():
    """'каменный меч' = материал + вид инструмента."""
    p = parse_command("сделай каменный меч")
    assert p["tier"] == "stone" and p["kind"] == "sword"
    assert p["target"] == "stone_sword"


def test_unknown_command_returns_none():
    """Непонятную команду агент не выдумывает, а честно не понимает."""
    assert command_to_goal("спой песню про кота", GOALS) is None


def test_count_is_parsed():
    assert parse_command("добудь 5 брёвен")["count"] == "5"


# --- честность ответов ----------------------------------------------------
def test_reply_reports_real_health(env):
    """Число в ответе совпадает с наблюдением — выдумывать нечем."""
    env.reset(goal_index=GOALS.index("wooden_sword"))
    env.agent.health = 5.0
    text = describe(env.observe(), "wooden_sword")
    assert "5 из 20" in text, text


def test_reply_prioritises_danger(env):
    """Смертельная опасность важнее задачи по крафту."""
    env.reset(goal_index=GOALS.index("wooden_sword"))
    env.agent.health = 3.0
    assert "опасно" in describe(env.observe(), "wooden_sword")


def test_reply_counts_remaining(env):
    """Сколько осталось — считается, а не берётся с потолка."""
    env.reset(goal_index=GOALS.index("oak_log"))
    env.agent.health = 20.0
    env.agent.food = 20.0
    text = describe(env.observe(), "oak_log", {"have": 1, "need": 4})
    assert "3" in text, text


def test_no_combat_talk_without_enemy(env):
    """
    Агент не должен рапортовать о драке, стоя один в поле.

    Это и есть та самая галлюцинация в миниатюре: фраза, не подтверждённая
    наблюдением. Ловим её тестом.
    """
    env.reset(goal_index=GOALS.index("oak_log"))
    env.mobs = []
    env.agent.health = 20.0
    env.agent.food = 20.0
    text = describe(env.observe(), "oak_log")
    assert "дерусь" not in text and "убегаю" not in text, text


# --- модель ---------------------------------------------------------------
def test_head_is_small():
    """Языковая голова должна быть на порядки меньше зрения."""
    m = LanguageHead(LangConfig())
    assert m.n_params() < 300_000, f"{m.n_params():,} параметров"


def test_head_forward_shapes(env):
    env.reset()
    m = LanguageHead(LangConfig())
    tok = torch.tensor([encode("сделай каменный меч")])
    dense = torch.tensor(env.observe()["dense"]).unsqueeze(0)
    goal_logits, reply_logits = m(tok, dense)
    assert goal_logits.shape[0] == 1 and reply_logits.shape[0] == 1


def test_encode_decode_roundtrip():
    text = "сделай каменный меч"
    assert decode(encode(text)) == text
