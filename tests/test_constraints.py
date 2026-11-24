"""配准约束位图、增量维护与旧库补建的测试。"""
from __future__ import annotations

import pytest

from wafer_review.constraints import (
    backfill_legacy_databases,
    is_built,
    load_pair_masks,
)
from wafer_review.db import connect, init_db
from wafer_review.geometry import ALL_TRANSFORMS, apply_transform
from wafer_review.model import (
    PAIR_MASK_ALL,
    Locator,
    Step,
    Wafer,
    compute_feasibility,
    derive_pair_masks,
    locator_pair_mask,
    pair_bit,
)
from wafer_review.service import (
    StepInput,
    add_locator,
    correct_locator,
    create_wafer,
    load_wafer,
    remove_locator,
)

SAMPLES = [
    ("光刻", "/workspace/samples/1_光刻.csv"),
    ("刻蚀", "/workspace/samples/2_刻蚀.csv"),
    ("沉积", "/workspace/samples/3_沉积.csv"),
    ("研磨", "/workspace/samples/4_研磨.csv"),
    ("检测", "/workspace/samples/5_检测.csv"),
    ("封装", "/workspace/samples/6_封装.csv"),
]


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test.sqlite3"
    init_db(path)
    return path


def sample_steps():
    return [
        StepInput(name, open(path, encoding="utf-8").read()) for name, path in SAMPLES
    ]


def test_single_locator_mask_marks_exactly_coinciding_pairs():
    n = 8
    cell_a, cell_b = (2, 7), (3, 2)
    mask = locator_pair_mask(cell_a, cell_b, n)
    for ti, ta in enumerate(ALL_TRANSFORMS):
        for tj, tb in enumerate(ALL_TRANSFORMS):
            coincide = apply_transform(ta, *cell_a, n) == apply_transform(tb, *cell_b, n)
            assert bool(mask & pair_bit(ti, tj)) == coincide
    # 8x8 种搭配全覆盖检查
    assert 0 <= mask <= PAIR_MASK_ALL


def test_pair_effective_mask_is_and_of_locator_fragments():
    """一对工序的有效约束 = 各同号点位图按位与（含“互不包含”情形）。"""
    n = 8
    m_p = locator_pair_mask((1, 1), (1, 1), n)
    m_q = locator_pair_mask((1, 8), (1, 8), n)
    assert m_p != m_q
    assert m_p & ~m_q & PAIR_MASK_ALL       # 存在只被 P 允许的搭配
    assert m_q & ~m_p & PAIR_MASK_ALL       # 存在只被 Q 允许的搭配
    assert m_p & m_q                         # 也存在共同允许的搭配


def test_restoration_only_when_other_locator_allows():
    """撤去一个同号点：被它排除的搭配只有别的同号点也允许时才恢复。"""
    n = 8
    m_p = locator_pair_mask((1, 1), (1, 1), n)
    m_q = locator_pair_mask((1, 8), (1, 8), n)
    only_p = m_p & ~m_q & PAIR_MASK_ALL
    only_q = m_q & ~m_p & PAIR_MASK_ALL

    # 两点都在：交集
    assert (m_p & m_q) & only_p == 0
    assert (m_p & m_q) & only_q == 0
    # 撤 P 后只剩 Q：only_q 恢复，only_p 中被 Q 排除的永不恢复
    restored_after_remove_p = m_q
    assert restored_after_remove_p & only_q == only_q
    assert restored_after_remove_p & only_p == 0
    # 撤 Q 后只剩 P：对称
    assert m_p & only_p == only_p
    assert m_p & only_q == 0


def test_stored_masks_match_derived_masks(db):
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert is_built(conn, wafer_id)
        stored = load_pair_masks(conn, wafer)
    assert stored == derive_pair_masks(wafer)
    # 库内逐片段按位与也应等于有效约束
    with connect(db) as conn:
        frag_and: dict[tuple[int, int], int] = {}
        for row in conn.execute(
            "SELECT step_i, step_j, allowed_mask FROM pair_constraint WHERE wafer_id = ?",
            (wafer_id,),
        ):
            key = (row["step_i"], row["step_j"])
            frag_and[key] = frag_and.get(key, PAIR_MASK_ALL) & int(row["allowed_mask"], 16)
    index_by_id = {s.id: i for i, s in enumerate(wafer.steps)}
    canon = {(index_by_id[a], index_by_id[b]): m for (a, b), m in frag_and.items()}
    assert canon == stored


def test_enumeration_uses_maintained_constraints(db):
    """查询走库内维护的约束：篡改位图会直接改变方案数，证明枚举读的是表而非重算。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert compute_feasibility(wafer).scheme_count == 2
        # 把某个片段改成全 0（任何搭配都不允许），枚举立即无解
        conn.execute(
            "UPDATE pair_constraint SET allowed_mask = '0' WHERE wafer_id = ?",
            (wafer_id,),
        )
        wafer2 = load_wafer(conn, wafer_id)
        assert not compute_feasibility(wafer2).solvable


def test_remove_locator_rebuilds_only_touched_pid(db):
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        before = {
            (r["step_i"], r["step_j"], r["pid"]): r["allowed_mask"]
            for r in conn.execute(
                "SELECT step_i, step_j, pid, allowed_mask FROM pair_constraint WHERE wafer_id = ?",
                (wafer_id,),
            )
        }
        wafer = load_wafer(conn, wafer_id)
        c1 = next(loc for loc in wafer.steps[2].locators if loc.pid == "C1")
    remove_locator(db, wafer_id, c1.id)
    with connect(db) as conn:
        after = {
            (r["step_i"], r["step_j"], r["pid"]): r["allowed_mask"]
            for r in conn.execute(
                "SELECT step_i, step_j, pid, allowed_mask FROM pair_constraint WHERE wafer_id = ?",
                (wafer_id,),
            )
        }
        # 其它编号的片段一行都没变
        untouched = {k: v for k, v in before.items() if k[2] != "C1"}
        assert all(after.get(k) == v for k, v in untouched.items())
        # C1 片段只剩封装以外工序之间的（沉积-封装那对已消失）
        c1_rows = {k for k in after if k[2] == "C1"}
        assert all(k[0] != c1.step_id and k[1] != c1.step_id for k in c1_rows)
        # 约束仍与定位点推导一致
        wafer2 = load_wafer(conn, wafer_id)
        assert load_pair_masks(conn, wafer2) == derive_pair_masks(wafer2)


def test_second_removal_keeps_constraints_consistent(db):
    """连续撤点：每次撤完库内约束都与剩余定位点推导结果一致。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        c1 = next(loc for loc in wafer.steps[2].locators if loc.pid == "C1")
    remove_locator(db, wafer_id, c1.id)
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert compute_feasibility(wafer).scheme_count == 8
        l1 = next(loc for loc in wafer.steps[0].locators if loc.pid == "L1")
    remove_locator(db, wafer_id, l1.id)
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert load_pair_masks(conn, wafer) == derive_pair_masks(wafer)
        assert compute_feasibility(wafer).solvable
        # 约束 meta 的内置修订号与晶圆修订号同步（同事务提交）
        revision = conn.execute(
            "SELECT revision FROM wafer WHERE id = ?", (wafer_id,)
        ).fetchone()[0]
        built = conn.execute(
            "SELECT built_revision FROM constraint_meta WHERE wafer_id = ?", (wafer_id,)
        ).fetchone()[0]
        assert built == revision == 2


def test_correct_and_add_refresh_fragments(db):
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        c1 = next(loc for loc in wafer.steps[5].locators if loc.pid == "C1")
    correct_locator(db, wafer_id, c1.id, 4, 4)
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert load_pair_masks(conn, wafer) == derive_pair_masks(wafer)
        feas = compute_feasibility(wafer)
        assert feas.scheme_count == 2

    # 在研磨与封装上补同一个新编号 Z1：按各自一个可行变换取同一基准格的逆像，
    # 保证与既有约束相容，应产生新的跨工序片段并保持可解。
    from wafer_review.geometry import invert_transform

    grind, pack = wafer.steps[3], wafer.steps[5]
    feas = compute_feasibility(wafer)
    t_grind = feas.for_step(grind.id).transforms[0]
    t_pack = feas.for_step(pack.id).transforms[0]
    base_cell = (3, 3)
    add_locator(db, wafer_id, grind.id, "Z1", *invert_transform(t_grind, *base_cell, 8))
    add_locator(db, wafer_id, pack.id, "Z1", *invert_transform(t_pack, *base_cell, 8))
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert load_pair_masks(conn, wafer) == derive_pair_masks(wafer)
        pids = {r["pid"] for r in conn.execute(
            "SELECT DISTINCT pid FROM pair_constraint WHERE wafer_id = ?", (wafer_id,)
        )}
        assert "Z1" in pids
        assert compute_feasibility(wafer).solvable


def test_legacy_database_is_backfilled_on_first_use(db):
    """旧库：只有定位点、没有约束表内容，首次启动/写操作时自动补建。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    # 模拟旧库：清空约束并删掉 meta 行
    with connect(db) as conn:
        conn.execute("DELETE FROM pair_constraint WHERE wafer_id = ?", (wafer_id,))
        conn.execute("DELETE FROM constraint_meta WHERE wafer_id = ?", (wafer_id,))
        assert not is_built(conn, wafer_id)

    # 启动期补建
    n = backfill_legacy_databases(db)
    assert n == 1
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert is_built(conn, wafer_id)
        assert load_pair_masks(conn, wafer) == derive_pair_masks(wafer)

    # 再启动一次不应重复补建
    assert backfill_legacy_databases(db) == 0


def test_legacy_database_backfilled_inside_write_transaction(db):
    """即使启动补建没跑，写事务里的 ensure_constraints 也会补上。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    with connect(db) as conn:
        conn.execute("DELETE FROM pair_constraint WHERE wafer_id = ?", (wafer_id,))
        conn.execute("DELETE FROM constraint_meta WHERE wafer_id = ?", (wafer_id,))
        wafer = load_wafer(conn, wafer_id)
        l2 = next(loc for loc in wafer.steps[0].locators if loc.pid == "L2")
    # 撤点事务内部先补建再增量维护，最终约束仍与定位点一致
    remove_locator(db, wafer_id, l2.id)
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert is_built(conn, wafer_id)
        assert load_pair_masks(conn, wafer) == derive_pair_masks(wafer)


def test_no_solution_rolls_back_locators_and_constraints_together(db):
    """无解更正整体回滚：定位点与配准约束都保持上一版提交的状态。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())

    def snapshot():
        with connect(db) as conn:
            locs = conn.execute(
                "SELECT step_id, pid, row, col FROM locator "
                "JOIN step ON locator.step_id = step.id WHERE step.wafer_id = ? ORDER BY locator.id",
                (wafer_id,),
            ).fetchall()
            frags = conn.execute(
                "SELECT step_i, step_j, pid, allowed_mask FROM pair_constraint "
                "WHERE wafer_id = ? ORDER BY id", (wafer_id,)
            ).fetchall()
            meta = conn.execute(
                "SELECT built_revision FROM constraint_meta WHERE wafer_id = ?", (wafer_id,)
            ).fetchone()
        return (
            [tuple(r) for r in locs],
            [tuple(r) for r in frags],
            meta["built_revision"],
        )

    before = snapshot()
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        c1 = wafer.steps[5].locators[0]
    from wafer_review.model import NoSolutionError

    with pytest.raises(NoSolutionError):
        correct_locator(db, wafer_id, c1.id, 1, 1)
    assert snapshot() == before


def test_refresh_uses_only_touched_pid_locator_rows(db, monkeypatch):
    """补/改/撤点时只从定位点明细读取本次涉及编号的记录。"""
    wafer_id = create_wafer(db, "LOT", 8, sample_steps())
    seen_pids: list[set[str]] = []
    import wafer_review.constraints as constraints_mod

    orig = constraints_mod._locators_by_pid

    def spy(conn, step, pids):
        seen_pids.append(set(pids))
        return orig(conn, step, pids)

    monkeypatch.setattr(constraints_mod, "_locators_by_pid", spy)
    with connect(db) as conn:
        wafer = load_wafer(conn, wafer_id)
        c1 = next(loc for loc in wafer.steps[2].locators if loc.pid == "C1")
    remove_locator(db, wafer_id, c1.id)
    assert seen_pids, "增量维护应当查询定位点明细"
    assert all(pids == {"C1"} for pids in seen_pids)


def test_in_memory_wafer_falls_back_to_derived_constraints():
    """没有库内约束的内存晶圆（如单测构造）仍可由定位点即时推导，行为不变。"""
    base = Step(1, 1, "基准", [Locator(1, "L1", 2, 7), Locator(2, "L2", 2, 6)], [])
    other = Step(
        2, 2, "工序2",
        [Locator(3, "L1", 2, 7), Locator(4, "L2", 2, 6), Locator(5, "C1", 4, 5)],
        [],
    )
    wafer = Wafer(1, "内存片", 8, 0, "", "", [base, other])
    assert wafer.pair_masks == {}
    feas = compute_feasibility(wafer)
    assert feas.solvable
    assert feas.steps[1].transforms == ((False, 0),)
