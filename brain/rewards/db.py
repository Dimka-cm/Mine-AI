"""
База данных наград (SQLite).

Отвечает ровно за то, о чём просил пользователь:
  * хранит ПРАВИЛА наград (за что и сколько давать/отнимать);
  * помнит, сколько раз агент уже получал каждую награду, и ЗАТУХАЕТ её,
    чтобы нельзя было фармить одно и то же действие бесконечно;
  * ведёт журнал всех начислений для дашборда и отладки.

Схема:
  reward_rules  — справочник правил (можно править на лету, без кода)
  reward_state  — счётчик выдач по каждому ключу (ключ = правило + контекст)
  reward_log    — история: шаг, эпизод, ключ, сколько дали, почему
  episodes      — сводка по эпизодам
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS reward_rules (
    key          TEXT PRIMARY KEY,   -- 'craft.stone_sword', 'grid.place_correct', ...
    category     TEXT NOT NULL,      -- craft / grid / spatial / penalty / milestone
    base_value   REAL NOT NULL,      -- награда за первое срабатывание
    decay        REAL NOT NULL DEFAULT 0.5,   -- множитель за каждое повторение
    min_value    REAL NOT NULL DEFAULT 0.0,   -- ниже этого не опускаемся
    max_repeats  INTEGER NOT NULL DEFAULT -1, -- -1 = без лимита выдач
    one_shot     INTEGER NOT NULL DEFAULT 0,  -- 1 = только один раз за всё время
    per_episode  INTEGER NOT NULL DEFAULT 0,  -- 1 = счётчик сбрасывается каждый эпизод
    description  TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reward_state (
    ctx_key      TEXT PRIMARY KEY,   -- 'craft.stone_sword' или 'grid.place_correct|slot=4'
    rule_key     TEXT NOT NULL,
    times        INTEGER NOT NULL DEFAULT 0,
    total_value  REAL NOT NULL DEFAULT 0.0,
    last_episode INTEGER NOT NULL DEFAULT -1,
    last_ts      REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (rule_key) REFERENCES reward_rules(key)
);

CREATE TABLE IF NOT EXISTS reward_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    episode   INTEGER NOT NULL,
    step      INTEGER NOT NULL,
    ctx_key   TEXT NOT NULL,
    rule_key  TEXT NOT NULL,
    value     REAL NOT NULL,
    times     INTEGER NOT NULL,
    reason    TEXT DEFAULT '',
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_ep ON reward_log(episode);
CREATE INDEX IF NOT EXISTS idx_log_rule ON reward_log(rule_key);

CREATE TABLE IF NOT EXISTS episodes (
    episode      INTEGER PRIMARY KEY,
    steps        INTEGER NOT NULL DEFAULT 0,
    total_reward REAL NOT NULL DEFAULT 0.0,
    positive     REAL NOT NULL DEFAULT 0.0,
    negative     REAL NOT NULL DEFAULT 0.0,
    crafted      TEXT NOT NULL DEFAULT '[]',
    started_ts   REAL NOT NULL DEFAULT 0.0,
    ended_ts     REAL NOT NULL DEFAULT 0.0
);
"""


@dataclass
class Rule:
    key: str
    category: str
    base_value: float
    decay: float = 0.5
    min_value: float = 0.0
    max_repeats: int = -1
    one_shot: bool = False
    per_episode: bool = False
    description: str = ""


@dataclass
class Grant:
    """Результат попытки выдать награду."""

    ctx_key: str
    rule_key: str
    value: float
    times: int
    reason: str


class RewardDB:
    """Тонкая обёртка над SQLite: правила + затухание + журнал."""

    def __init__(self, path: str | Path = "data/rewards.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + собственный лок: дашборд читает базу из
        # HTTP-потока, пока цикл обучения пишет в неё из своего.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self._rules: Dict[str, Rule] = {}
        self._load_rules()

    # ---------------- правила ----------------
    def _load_rules(self) -> None:
        self._rules = {}
        with self._lock:
            rows = list(self.conn.execute("SELECT * FROM reward_rules"))
        for row in rows:
            self._rules[row["key"]] = Rule(
                key=row["key"], category=row["category"],
                base_value=row["base_value"], decay=row["decay"],
                min_value=row["min_value"], max_repeats=row["max_repeats"],
                one_shot=bool(row["one_shot"]), per_episode=bool(row["per_episode"]),
                description=row["description"] or "",
            )

    def upsert_rule(self, rule: Rule) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO reward_rules
                   (key, category, base_value, decay, min_value, max_repeats,
                    one_shot, per_episode, description)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     category=excluded.category, base_value=excluded.base_value,
                     decay=excluded.decay, min_value=excluded.min_value,
                     max_repeats=excluded.max_repeats, one_shot=excluded.one_shot,
                     per_episode=excluded.per_episode, description=excluded.description
                """,
                (rule.key, rule.category, rule.base_value, rule.decay, rule.min_value,
                 rule.max_repeats, int(rule.one_shot), int(rule.per_episode),
                 rule.description),
            )
            self.conn.commit()
            self._rules[rule.key] = rule

    def upsert_many(self, rules: Iterable[Rule]) -> None:
        for r in rules:
            self.upsert_rule(r)

    def rule(self, key: str) -> Optional[Rule]:
        return self._rules.get(key)

    def all_rules(self) -> List[Rule]:
        return sorted(self._rules.values(), key=lambda r: (r.category, r.key))

    # ---------------- выдача награды ----------------
    def grant(self, rule_key: str, *, episode: int, step: int,
              ctx: str = "", reason: str = "", commit: bool = True) -> Grant:
        """
        Начисляет награду с учётом истории.

        ctx позволяет разделять счётчики: 'craft.tool' с ctx='stone_sword'
        затухает отдельно от ctx='iron_sword'. Именно так выполняется
        требование "для каменного/железного — так же, но заново".
        """
        rule = self._rules.get(rule_key)
        if rule is None:
            return Grant(rule_key, rule_key, 0.0, 0, f"нет правила {rule_key}")

        ctx_key = f"{rule_key}|{ctx}" if ctx else rule_key
        with self._lock:
            row = self.conn.execute(
                "SELECT times, last_episode FROM reward_state WHERE ctx_key=?",
                (ctx_key,)).fetchone()
        times = row["times"] if row else 0
        last_ep = row["last_episode"] if row else -1

        # Счётчик, который живёт только внутри эпизода.
        if rule.per_episode and last_ep != episode:
            times = 0

        if rule.one_shot and times >= 1:
            return Grant(ctx_key, rule_key, 0.0, times, "уже получено (one_shot)")
        if rule.max_repeats >= 0 and times >= rule.max_repeats:
            return Grant(ctx_key, rule_key, 0.0, times, "лимит повторов исчерпан")

        # ЗАТУХАНИЕ: value = base * decay^times, но не ниже min_value.
        value = rule.base_value * (rule.decay ** times)
        if rule.base_value >= 0:
            value = max(value, rule.min_value)
        else:
            # Штрафы не затухают до нуля — за ошибку бьём стабильно.
            value = min(value, -abs(rule.min_value)) if rule.min_value else value

        times += 1
        now = time.time()
        with self._lock:
            self.conn.execute(
                """INSERT INTO reward_state (ctx_key, rule_key, times,
                                             total_value, last_episode, last_ts)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(ctx_key) DO UPDATE SET
                     times=?, total_value=reward_state.total_value+?,
                     last_episode=?, last_ts=?""",
                (ctx_key, rule_key, times, value, episode, now,
                 times, value, episode, now),
            )
            self.conn.execute(
                """INSERT INTO reward_log (episode, step, ctx_key, rule_key,
                                           value, times, reason, ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (episode, step, ctx_key, rule_key, value, times,
                 reason or rule.description, now),
            )
            if commit:
                self.conn.commit()
        return Grant(ctx_key, rule_key, value, times, reason or rule.description)

    def peek(self, rule_key: str, ctx: str = "") -> Tuple[int, float]:
        """Сколько раз уже выдавали и какой будет следующая награда."""
        rule = self._rules.get(rule_key)
        if rule is None:
            return 0, 0.0
        ctx_key = f"{rule_key}|{ctx}" if ctx else rule_key
        with self._lock:
            row = self.conn.execute(
                "SELECT times FROM reward_state WHERE ctx_key=?", (ctx_key,)
            ).fetchone()
        times = row["times"] if row else 0
        if rule.one_shot and times >= 1:
            return times, 0.0
        if rule.max_repeats >= 0 and times >= rule.max_repeats:
            return times, 0.0
        val = rule.base_value * (rule.decay ** times)
        if rule.base_value >= 0:
            val = max(val, rule.min_value)
        return times, val

    # ---------------- эпизоды / статистика ----------------
    def start_episode(self, episode: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO episodes (episode, started_ts) VALUES (?,?)",
                (episode, time.time()),
            )
            self.conn.commit()

    def finish_episode(self, episode: int, steps: int, total: float,
                       positive: float, negative: float, crafted: List[str]) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE episodes SET steps=?, total_reward=?, positive=?, negative=?,
                                       crafted=?, ended_ts=? WHERE episode=?""",
                (steps, total, positive, negative, json.dumps(crafted), time.time(), episode),
            )
            self.conn.commit()

    def reset_progress(self, keep_rules: bool = True) -> None:
        """Забыть всю историю наград (модель учится 'с нуля')."""
        with self._lock:
            self.conn.executescript(
                "DELETE FROM reward_state; DELETE FROM reward_log; DELETE FROM episodes;"
            )
            if not keep_rules:
                self.conn.execute("DELETE FROM reward_rules")
                self._rules = {}
            self.conn.commit()

    def top_rules(self, limit: int = 20) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                """SELECT rule_key, ctx_key, times, total_value
                   FROM reward_state ORDER BY ABS(total_value) DESC LIMIT ?""", (limit,)
            ))

    def episode_stats(self, limit: int = 200) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                """SELECT * FROM (SELECT * FROM episodes WHERE ended_ts > 0
                   ORDER BY episode DESC LIMIT ?) ORDER BY episode ASC""", (limit,)
            ))

    def recent_log(self, limit: int = 60) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM reward_log ORDER BY id DESC LIMIT ?", (limit,)
            ))

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()
