"""SQLite 持久化层。

一次“上传 / 更正定位点 / 撤去定位点”都是一个事务：写入新的定位点后，在同一
事务里校验是否存在可行方案。若不存在则回滚，因此“无解或保存失败就沿用原定位
点和原结果”。查询结果不落库，总是由已保存的定位点即时推导，避免结果与定位点
不同步。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS wafer (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    grid_size   INTEGER NOT NULL CHECK (grid_size BETWEEN 2 AND 60),
    revision    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS step (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id    INTEGER NOT NULL REFERENCES wafer(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    UNIQUE (wafer_id, seq),
    UNIQUE (wafer_id, name)
);

CREATE TABLE IF NOT EXISTS locator (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    pid         TEXT    NOT NULL,
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    UNIQUE (step_id, pid)
);

CREATE TABLE IF NOT EXISTS defect (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    did         TEXT    NOT NULL,
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    UNIQUE (step_id, did)
);

-- 配准约束的最小片段：某个同号定位点对某一对工序允许的“变换搭配”位图。
-- allowed_mask 为 64 位整数，位 k*8+m 置 1 表示“第 i 道用变换 k、第 j 道用
-- 变换 m”时该同号点换算后重合。一对工序的有效约束 = 各同号点位图按位与，
-- 因此撤去一个同号点只需删掉它自己的片段。
CREATE TABLE IF NOT EXISTS pair_constraint (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id     INTEGER NOT NULL REFERENCES wafer(id) ON DELETE CASCADE,
    step_i       INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    step_j       INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    pid          TEXT    NOT NULL,
    -- 64 位位图可能超过 SQLite 有符号 int64 上限，统一存十六进制文本。
    allowed_mask TEXT    NOT NULL,
    UNIQUE (wafer_id, step_i, step_j, pid)
);

-- 每张晶圆一行：约束已随哪一版定位点建好。旧库首次使用时若缺行则补建。
CREATE TABLE IF NOT EXISTS constraint_meta (
    wafer_id      INTEGER PRIMARY KEY REFERENCES wafer(id) ON DELETE CASCADE,
    built_revision INTEGER NOT NULL,
    built_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_wafer ON step(wafer_id, seq);
CREATE INDEX IF NOT EXISTS idx_locator_step ON locator(step_id);
CREATE INDEX IF NOT EXISTS idx_defect_step ON defect(step_id);
CREATE INDEX IF NOT EXISTS idx_pair_wafer ON pair_constraint(wafer_id, step_i, step_j);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
