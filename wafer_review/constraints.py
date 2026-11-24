"""配准约束的持久化与增量维护。

约束的最小片段是“工序对 × 同号定位点编号”的 64 位变换搭配位图（见
``model.locator_pair_mask``）。表结构与定位点同库同事务维护：

* 建片：定位点写入后一次性建好全部片段与 ``constraint_meta``；
* 补点/改点/撤点：只读取本次涉及编号在各工序上的定位点明细，删掉这些编号的
  旧片段后按当前同号点整体重建——其它编号、其它工序对的行完全不碰；
* 撤点后含被撤工序的配对自然不再出现。一对工序的有效约束是各同号点位图的
  按位与，所以原先被撤去点排除的变换搭配，只有在别的同号点也允许（即按位与
  后仍为 1）时才恢复；
* 旧库首次使用：``ensure_constraints`` 发现缺 ``constraint_meta`` 行就按当前
  定位点全量补建。

所有函数都接收调用方事务内的连接，不自行提交，保证“定位点与约束一起生效、
一次查询读到同一次提交的状态”。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .db import connect, utcnow
from .model import (
    PAIR_MASK_ALL,
    Locator,
    Step,
    Wafer,
    locator_pair_mask,
)


def load_pair_masks(
    conn: sqlite3.Connection, wafer: Wafer
) -> dict[tuple[int, int], int]:
    """读已维护的有效工序对约束（片段按位与），键为 steps 下标对 (i, j)。"""
    if not wafer.steps:
        return {}
    index_by_step_id = {step.id: idx for idx, step in enumerate(wafer.steps)}
    masks: dict[tuple[int, int], int] = {}
    rows = conn.execute(
        "SELECT step_i, step_j, allowed_mask FROM pair_constraint WHERE wafer_id = ?",
        (wafer.id,),
    )
    for row in rows:
        i = index_by_step_id.get(row["step_i"])
        j = index_by_step_id.get(row["step_j"])
        if i is None or j is None:
            # 理论上不会发生（外键级联），防御性跳过。
            continue
        key = (min(i, j), max(i, j))
        masks[key] = masks.get(key, PAIR_MASK_ALL) & int(row["allowed_mask"], 16)
    return masks


def is_built(conn: sqlite3.Connection, wafer_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM constraint_meta WHERE wafer_id = ?", (wafer_id,)
    ).fetchone()
    return row is not None


def _replace_meta(conn: sqlite3.Connection, wafer_id: int, revision: int) -> None:
    conn.execute(
        "INSERT INTO constraint_meta (wafer_id, built_revision, built_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(wafer_id) DO UPDATE SET built_revision = excluded.built_revision, "
        "built_at = excluded.built_at",
        (wafer_id, revision, utcnow()),
    )


def build_all_constraints(conn: sqlite3.Connection, wafer: Wafer) -> None:
    """按当前定位点全量重建约束（建片与旧库补建使用）。"""
    conn.execute("DELETE FROM pair_constraint WHERE wafer_id = ?", (wafer.id,))
    # 编号 -> [(工序下标, 定位点), ...]
    points: dict[str, list[tuple[int, Locator]]] = {}
    for idx, step in enumerate(wafer.steps):
        for loc in step.locators:
            points.setdefault(loc.pid, []).append((idx, loc))

    rows: list[tuple[int, int, int, str, str]] = []
    n = wafer.grid_size
    for pid, entries in points.items():
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                idx_a, loc_a = entries[a]
                idx_b, loc_b = entries[b]
                step_i, step_j = wafer.steps[idx_a], wafer.steps[idx_b]
                mask = locator_pair_mask((loc_a.row, loc_a.col), (loc_b.row, loc_b.col), n)
                rows.append((wafer.id, step_i.id, step_j.id, pid, format(mask, "x")))
    if rows:
        conn.executemany(
            "INSERT INTO pair_constraint "
            "(wafer_id, step_i, step_j, pid, allowed_mask) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    _replace_meta(conn, wafer.id, wafer.revision)


def ensure_constraints(conn: sqlite3.Connection, wafer: Wafer) -> Wafer:
    """旧库首次使用时补建约束，并把有效位图挂到晶圆对象上返回。

    补建只在调用方事务内进行、不单独提交；随后的写操作若失败会整体回滚，
    不会留下半截约束。
    """
    if not is_built(conn, wafer.id):
        build_all_constraints(conn, wafer)
    wafer.pair_masks = load_pair_masks(conn, wafer)
    return wafer


def backfill_legacy_databases(db_path: str | Path) -> int:
    """应用启动时调用：给旧库里没有约束记录的晶圆逐个补建。

    每张晶圆一个独立事务，补建失败不影响其它晶圆；该晶圆在下一次写操作时
    还会由 ``ensure_constraints`` 再试一次。返回补建的晶圆数。
    """
    backfilled = 0
    with connect(db_path) as conn:
        wafer_ids = [r["id"] for r in conn.execute("SELECT id FROM wafer ORDER BY id")]
        for wafer_id in wafer_ids:
            if is_built(conn, wafer_id):
                continue
            # 延迟导入避免与 service 的模块级依赖形成环。
            from .service import load_wafer

            wafer = load_wafer(conn, wafer_id)
            if wafer is None:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                build_all_constraints(conn, wafer)
                conn.execute("COMMIT")
                backfilled += 1
            except Exception:
                conn.execute("ROLLBACK")
                raise
    return backfilled


def _locators_by_pid(conn: sqlite3.Connection, step: Step, pids: set[str]) -> dict[str, Locator]:
    """只读取本次涉及编号的定位点明细（校验不再扫全部同号点）。"""
    if not pids:
        return {}
    placeholders = ",".join("?" for _ in pids)
    rows = conn.execute(
        f"SELECT id, pid, row, col FROM locator WHERE step_id = ? "
        f"AND pid IN ({placeholders})",
        (step.id, *sorted(pids)),
    )
    return {r["pid"]: Locator(r["id"], r["pid"], r["row"], r["col"]) for r in rows}


def refresh_fragments_for_pids(
    conn: sqlite3.Connection,
    wafer: Wafer,
    touched_pids: set[str],
) -> None:
    """补点/改点/撤点后，增量重建涉及编号的全部工序对片段。

    对每个涉及编号：先删掉它在 ``pair_constraint`` 里的全部片段，再按库里
    当前剩余的同号点明细（已含调用方在本事务里的写入）整体重建。撤点时含被
    撤工序的配对自然不再出现；一对工序的有效约束在读取侧把剩余同号点的片段
    按位与，所以原先被撤去点排除的变换搭配，只有别的同号点也允许时才恢复。
    只查 ``touched_pids`` 的定位点记录，不触碰无关编号。
    """
    if not touched_pids:
        return
    n = wafer.grid_size
    # 每个涉及编号在各工序上的当前明细（已含调用方在本事务里的写入）。
    per_step: list[dict[str, Locator]] = [
        _locators_by_pid(conn, step, touched_pids) for step in wafer.steps
    ]

    for pid in touched_pids:
        # 先删掉该编号的全部片段，再按库里当前的同号点整体重建：撤点场景下
        # 含被撤工序的配对自然不会再出现，而剩余工序之间的片段以新计算值为准
        # （多个同号点时，按位与关系由各编号片段在读取侧合成）。
        conn.execute(
            "DELETE FROM pair_constraint WHERE wafer_id = ? AND pid = ?",
            (wafer.id, pid),
        )
        present = [
            (idx, locs[pid]) for idx, locs in enumerate(per_step) if pid in locs
        ]
        rows: list[tuple[int, int, int, str, str]] = []
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                idx_a, loc_a = present[a]
                idx_b, loc_b = present[b]
                mask = locator_pair_mask((loc_a.row, loc_a.col), (loc_b.row, loc_b.col), n)
                rows.append(
                    (wafer.id, wafer.steps[idx_a].id, wafer.steps[idx_b].id, pid, format(mask, "x"))
                )
        if rows:
            conn.executemany(
                "INSERT INTO pair_constraint "
                "(wafer_id, step_i, step_j, pid, allowed_mask) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    _replace_meta(conn, wafer.id, wafer.revision)
