"""领域模型与坐标查询。

约束：为每道工序选择一个 D4 变换（基准图固定为恒等变换），使得在所有工序中
出现过的同一个定位点编号，经各自变换换算到基准坐标系后落在同一个格子上。

某个编号若在基准图里也有，这个公共格就是基准图上的对应定位点；若基准图没有
该编号，则只要其余各图中同号点重合即可，公共格可以是任意一格。

配准约束按**工序对 + 定位点编号**持久化（见 ``constraints.py`` 与表
``pair_constraint``）：一个同号点允许一对工序使用的全部“变换搭配”记成一个
64 位位图（8×8=64 种搭配），一对工序的有效约束是该对所有同号点位图的按位与。
因此撤去一个同号点只删除它自己的片段，原先被排除的搭配只有在别的同号点也
允许时才恢复。校验与查询都直接枚举这些已维护的约束，不再逐个比对同号点。

由于不同工序可能共享“基准图没有的同号定位点”，各工序的变换会互相牵制，
所以这里枚举的是全局方案；某工序某缺陷的可能落点，就是它在所有可行方案下
换算结果的集合。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .geometry import (
    ALL_TRANSFORMS,
    TRANSFORM_COUNT,
    Transform,
    apply_transform,
    transform_labels,
)

# 一对工序最多 8×8 种变换搭配；位图 0 位对应 (变换0, 变换0)，以此类推。
PAIR_MASK_ALL = (1 << (TRANSFORM_COUNT * TRANSFORM_COUNT)) - 1  # 64 位全 1

class WaferError(Exception):
    """可直接展示给用户的业务错误。"""


class ValidationError(WaferError):
    pass


class NoSolutionError(WaferError):
    pass


@dataclass
class Locator:
    id: int
    pid: str
    row: int
    col: int


@dataclass
class Defect:
    id: int
    did: str
    row: int
    col: int


@dataclass
class Step:
    id: int
    seq: int
    name: str
    locators: list[Locator] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)

    @property
    def is_baseline(self) -> bool:
        return self.seq == 1

    def locator_map(self) -> dict[str, tuple[int, int]]:
        return {loc.pid: (loc.row, loc.col) for loc in self.locators}


@dataclass
class Wafer:
    id: int
    name: str
    grid_size: int
    revision: int
    created_at: str
    updated_at: str
    steps: list[Step] = field(default_factory=list)
    # 已维护的工序对约束：{(i, j): 64 位位图}，i/j 是 steps 下标，i < j。
    # 为空且 steps 多于 1 道时，回退到由定位点即时推导（单测构造的内存晶圆走这条路）。
    pair_masks: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def baseline(self) -> Step:
        return self.steps[0]

    def step_by_id(self, step_id: int) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def invalidate_constraints(self) -> None:
        """内存里的定位点被直接改动后调用：丢弃已维护约束，改由定位点即时推导。

        正常的补点/改点/撤点都走服务层事务并同步重建库内约束，不需要调它；
        这是给测试或脚本里“直接改 Wafer.steps”的场景准备的安全阀。
        """
        self.pair_masks = {}


@dataclass(frozen=True)
class DefectResult:
    """一次“工序 + 缺陷号”查询的结果。"""

    step_id: int
    step_name: str
    did: str
    source_row: int
    source_col: int
    status: str  # determined | pending | no_solution
    coordinate: tuple[int, int] | None
    candidates: tuple[tuple[int, int], ...]
    feasible_transforms: tuple[Transform, ...]
    scheme_count: int


@dataclass(frozen=True)
class StepFeasibility:
    step_id: int
    transforms: tuple[Transform, ...]
    labels: tuple[str, ...]
    shared_locators: tuple[str, ...]
    baseline_anchors: tuple[str, ...]
    unanchored_locators: tuple[str, ...]

    @property
    def is_ambiguous(self) -> bool:
        return len(self.transforms) > 1


@dataclass(frozen=True)
class Feasibility:
    steps: tuple[StepFeasibility, ...]
    solvable: bool
    scheme_count: int
    elapsed_ms: float = 0.0

    def for_step(self, step_id: int) -> StepFeasibility | None:
        return next((s for s in self.steps if s.step_id == step_id), None)


# --------------------------------------------------------------------------- #
# 配准约束（变换搭配位图）
# --------------------------------------------------------------------------- #

def pair_bit(ti: int, tj: int) -> int:
    """一对工序分别取第 ti、tj 号变换时对应的二进制位。"""
    return 1 << (ti * TRANSFORM_COUNT + tj)


def locator_pair_mask(
    cell_a: tuple[int, int], cell_b: tuple[int, int], n: int
) -> int:
    """单个同号定位点对一对工序产生的约束位图。

    置 1 的位表示“这两种变换搭配会把两个同号点换算到基准坐标系的同一格”。
    """
    mask = 0
    for ti, ta in enumerate(ALL_TRANSFORMS):
        image_a = apply_transform(ta, cell_a[0], cell_a[1], n)
        for tj, tb in enumerate(ALL_TRANSFORMS):
            if apply_transform(tb, cell_b[0], cell_b[1], n) == image_a:
                mask |= pair_bit(ti, tj)
    return mask


def derive_pair_masks(wafer: Wafer) -> dict[tuple[int, int], int]:
    """由当前定位点逐编号推导所有工序对的有效约束（按位与后的位图）。

    既用于首次建片/旧库补建，也用于校验“数据库里维护的约束与定位点一致”。
    """
    points: dict[str, list[tuple[int, tuple[int, int]]]] = {}
    for idx, step in enumerate(wafer.steps):
        for loc in step.locators:
            points.setdefault(loc.pid, []).append((idx, (loc.row, loc.col)))

    pair_masks: dict[tuple[int, int], int] = {}
    for entries in points.values():
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                i, cell_i = entries[a]
                j, cell_j = entries[b]
                if i > j:
                    i, j = j, i
                    cell_i, cell_j = cell_j, cell_i
                bit = locator_pair_mask(cell_i, cell_j, wafer.grid_size)
                key = (i, j)
                pair_masks[key] = pair_masks.get(key, PAIR_MASK_ALL) & bit
    return pair_masks


def _effective_pair_masks(wafer: Wafer) -> dict[tuple[int, int], int]:
    """枚举时实际使用的工序对约束：优先已维护的，缺失则即时推导。"""
    if wafer.pair_masks:
        return wafer.pair_masks
    if len(wafer.steps) > 1:
        return derive_pair_masks(wafer)
    return {}


def _pair_allowed(pair_masks: dict[tuple[int, int], int], i: int, j: int,
                  ti: int, tj: int) -> bool:
    """没有共享定位点的工序对之间没有约束（任何搭配都允许）。"""
    if i > j:
        i, j = j, i
        ti, tj = tj, ti
    mask = pair_masks.get((i, j))
    if mask is None:
        return True
    return bool(mask & pair_bit(ti, tj))


# --------------------------------------------------------------------------- #
# 全局方案枚举
# --------------------------------------------------------------------------- #

def _involved_locators(wafer: Wafer) -> list[frozenset[str]]:
    """逐工序收集真正参与跨工序约束的同号定位点编号（页面提示用）。"""
    points: dict[str, list[int]] = {}
    for idx, step in enumerate(wafer.steps):
        for loc in step.locators:
            points.setdefault(loc.pid, []).append(idx)
    involved: list[set[str]] = [set() for _ in wafer.steps]
    for pid, idxs in points.items():
        if len(idxs) > 1:
            for i in idxs:
                involved[i].add(pid)
    return [frozenset(s) for s in involved]


def enumerate_masks(
    wafer: Wafer,
) -> tuple[int, list[set[Transform]], list[frozenset[str]], float, dict[tuple[int, int], int]]:
    """基于已维护的工序对位图枚举全部可行全局方案。

    返回方案数、逐工序可行变换投影、逐工序参与约束的编号、耗时，以及本次
    实际使用的工序对位图（便于与持久化内容核对）。
    """
    start = time.perf_counter()
    steps = wafer.steps
    if not steps:
        return 0, [], [], 0.0, {}
    pair_masks = _effective_pair_masks(wafer)
    involved = _involved_locators(wafer)

    projections: list[set[Transform]] = [set() for _ in steps]
    chosen: list[int] = []
    scheme_count = 0

    def consistent(i: int, ti: int) -> bool:
        for j in range(i):
            if not _pair_allowed(pair_masks, i, j, ti, chosen[j]):
                return False
        return True

    def walk(i: int) -> None:
        nonlocal scheme_count
        if i == len(steps):
            scheme_count += 1
            for k, idx in enumerate(chosen):
                projections[k].add(ALL_TRANSFORMS[idx])
            return
        # 基准图固定恒等变换（编号 0），其余工序在 8 种 D4 变换里选。
        candidates = (0,) if i == 0 else range(TRANSFORM_COUNT)
        for ti in candidates:
            if consistent(i, ti):
                chosen.append(ti)
                walk(i + 1)
                chosen.pop()

    walk(0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scheme_count, projections, involved, elapsed_ms, pair_masks


def _enumerate(wafer: Wafer) -> tuple[int, list[set[Transform]], list[frozenset[str]], float]:
    """兼容旧调用：位图枚举的四元组版本。"""
    scheme_count, projections, involved, elapsed_ms, _ = enumerate_masks(wafer)
    return scheme_count, projections, involved, elapsed_ms


def _build_feasibility(
    wafer: Wafer,
    scheme_count: int,
    projections: list[set[Transform]],
    involved: list[frozenset[str]],
    elapsed_ms: float,
) -> Feasibility:
    base_ids = set(wafer.baseline.locator_map())
    step_feas: list[StepFeasibility] = []
    for idx, step in enumerate(wafer.steps):
        own = set(step.locator_map())
        constrained = own & involved[idx]
        transforms = tuple(t for t in ALL_TRANSFORMS if t in projections[idx])
        step_feas.append(
            StepFeasibility(
                step_id=step.id,
                transforms=transforms,
                labels=tuple(transform_labels(transforms)),
                shared_locators=tuple(sorted(constrained)),
                baseline_anchors=tuple(sorted(constrained & base_ids)),
                unanchored_locators=tuple(sorted(constrained - base_ids)),
            )
        )
    solvable = scheme_count > 0 and all(s.transforms for s in step_feas)
    return Feasibility(
        steps=tuple(step_feas),
        solvable=solvable,
        scheme_count=scheme_count,
        elapsed_ms=elapsed_ms,
    )


def compute_feasibility(wafer: Wafer) -> Feasibility:
    scheme_count, projections, involved, elapsed_ms = _enumerate(wafer)
    return _build_feasibility(wafer, scheme_count, projections, involved, elapsed_ms)


class Analysis:
    """在一次请求内复用枚举结果，避免为每个缺陷重复搜索。"""

    def __init__(self, wafer: Wafer) -> None:
        self.wafer = wafer
        scheme_count, projections, involved, elapsed_ms, self.pair_masks = enumerate_masks(wafer)
        self.feasibility = _build_feasibility(
            wafer, scheme_count, projections, involved, elapsed_ms
        )
        self._transforms = {s.step_id: s.transforms for s in self.feasibility.steps}

    @property
    def solvable(self) -> bool:
        return self.feasibility.solvable

    def transforms(self, step: Step) -> tuple[Transform, ...]:
        return self._transforms.get(step.id, ())

    def result(self, step: Step, defect: Defect) -> DefectResult:
        transforms = self.transforms(step)
        if not self.feasibility.solvable or not transforms:
            return DefectResult(
                step_id=step.id,
                step_name=step.name,
                did=defect.did,
                source_row=defect.row,
                source_col=defect.col,
                status="no_solution",
                coordinate=None,
                candidates=(),
                feasible_transforms=(),
                scheme_count=0,
            )
        if step.is_baseline:
            candidates = {(defect.row, defect.col)}
        else:
            candidates = {
                apply_transform(t, defect.row, defect.col, self.wafer.grid_size)
                for t in transforms
            }
        ordered = tuple(sorted(candidates))
        determined = len(ordered) == 1
        return DefectResult(
            step_id=step.id,
            step_name=step.name,
            did=defect.did,
            source_row=defect.row,
            source_col=defect.col,
            status="determined" if determined else "pending",
            coordinate=ordered[0] if determined else None,
            candidates=ordered,
            feasible_transforms=transforms,
            scheme_count=self.feasibility.scheme_count,
        )

    def step_results(self, step: Step) -> list[DefectResult]:
        return [self.result(step, d) for d in step.defects]

    def query(self, step: Step, did: str) -> DefectResult | None:
        defect = next((d for d in step.defects if d.did == did), None)
        return self.result(step, defect) if defect is not None else None


def analyze(wafer: Wafer) -> Analysis:
    return Analysis(wafer)


def query_defect(wafer: Wafer, step_id: int, did: str) -> DefectResult:
    step = wafer.step_by_id(step_id)
    if step is None:
        raise ValidationError("工序不存在")
    result = analyze(wafer).query(step, did)
    if result is None:
        raise ValidationError(f"工序“{step.name}”中没有缺陷 {did}")
    return result


def step_defect_results(wafer: Wafer, step_id: int) -> list[DefectResult]:
    step = wafer.step_by_id(step_id)
    if step is None:
        raise ValidationError("工序不存在")
    return analyze(wafer).step_results(step)
