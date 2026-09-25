"""Hierarchical, window-free evidence regions for EviViT Stage G.

The module builds a max-tree from a two-dimensional, question-conditioned
PTEA evidence map.  Pixels are activated from high to low evidence.  Every
new connected component becomes a leaf and every spatial merge/expansion
becomes one of its ancestors.  Consequently, region extent is determined by
the topology of the evidence map rather than by a catalogue of fixed crop
sizes.

The implementation intentionally contains no human annotation logic.  Human
traces are consumed by the separate Stage-G dataset builder and are never
needed at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from evivit_core.dense_evidence import normalize_distribution


@dataclass(frozen=True)
class EvidenceComponentNode:
    """One connected upper-level set in an evidence max-tree.

    ``grid_box`` is half open: ``(x0, y0, x1, y1)``.  ``cell_count`` is the
    actual component support, whereas the box may also contain holes.  The
    distinction is useful because the deployed crop must be rectangular but
    the evidence support need not be.
    """

    node_id: int
    parent_id: int | None
    child_ids: tuple[int, ...]
    level: float
    peak: float
    peak_grid: tuple[int, int]
    grid_box: tuple[int, int, int, int]
    cell_count: int
    component_mass: float
    component_energy: float
    component_entropy_sum: float
    map_height: int
    map_width: int

    @property
    def threshold_ratio(self) -> float:
        return self.level / max(self.peak, 1e-12)

    @property
    def area_ratio(self) -> float:
        x0, y0, x1, y1 = self.grid_box
        return (x1 - x0) * (y1 - y0) / float(self.map_height * self.map_width)

    @property
    def fill_ratio(self) -> float:
        x0, y0, x1, y1 = self.grid_box
        box_cells = max(1, (x1 - x0) * (y1 - y0))
        return self.cell_count / float(box_cells)

    @property
    def component_density(self) -> float:
        return self.component_mass / max(self.cell_count, 1)

    @property
    def aspect_ratio(self) -> float:
        x0, y0, x1, y1 = self.grid_box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        return max(width / height, height / width)

    @property
    def policy_box(self) -> list[int]:
        x0, y0, x1, y1 = self.grid_box
        return [
            int(round(1000 * x0 / self.map_width)),
            int(round(1000 * y0 / self.map_height)),
            int(round(1000 * x1 / self.map_width)),
            int(round(1000 * y1 / self.map_height)),
        ]


@dataclass(frozen=True)
class EvidenceComponentTree:
    """A complete max-tree and its inference-facing pruned candidate view."""

    nodes: tuple[EvidenceComponentNode, ...]
    root_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    map_height: int
    map_width: int

    def node(self, node_id: int) -> EvidenceComponentNode:
        return self.nodes[node_id]

    @property
    def candidates(self) -> tuple[EvidenceComponentNode, ...]:
        return tuple(self.nodes[node_id] for node_id in self.candidate_ids)


class _DisjointSet:
    def __init__(self, values: np.ndarray) -> None:
        count = values.size
        self.parent = np.full(count, -1, dtype=np.int64)
        self.size = np.zeros(count, dtype=np.int64)
        self.mass = np.zeros(count, dtype=np.float64)
        self.energy = np.zeros(count, dtype=np.float64)
        self.entropy_sum = np.zeros(count, dtype=np.float64)
        self.x0 = np.zeros(count, dtype=np.int64)
        self.y0 = np.zeros(count, dtype=np.int64)
        self.x1 = np.zeros(count, dtype=np.int64)
        self.y1 = np.zeros(count, dtype=np.int64)
        self.peak = np.zeros(count, dtype=np.float64)
        self.peak_index = np.full(count, -1, dtype=np.int64)
        self.values = values.reshape(-1)

    def active(self, index: int) -> bool:
        return self.parent[index] >= 0

    def activate(self, index: int, *, x: int, y: int) -> None:
        value = float(self.values[index])
        self.parent[index] = index
        self.size[index] = 1
        self.mass[index] = value
        self.energy[index] = value * value
        self.entropy_sum[index] = -value * np.log(max(value, 1e-30))
        self.x0[index], self.x1[index] = x, x + 1
        self.y0[index], self.y1[index] = y, y + 1
        self.peak[index] = value
        self.peak_index[index] = index

    def find(self, index: int) -> int:
        parent = int(self.parent[index])
        while parent != int(self.parent[parent]):
            self.parent[parent] = self.parent[int(self.parent[parent])]
            parent = int(self.parent[parent])
        while index != parent:
            next_index = int(self.parent[index])
            self.parent[index] = parent
            index = next_index
        return parent

    def union(self, first: int, second: int) -> int:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return first_root
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        self.mass[first_root] += self.mass[second_root]
        self.energy[first_root] += self.energy[second_root]
        self.entropy_sum[first_root] += self.entropy_sum[second_root]
        self.x0[first_root] = min(self.x0[first_root], self.x0[second_root])
        self.y0[first_root] = min(self.y0[first_root], self.y0[second_root])
        self.x1[first_root] = max(self.x1[first_root], self.x1[second_root])
        self.y1[first_root] = max(self.y1[first_root], self.y1[second_root])
        if self.peak[second_root] > self.peak[first_root]:
            self.peak[first_root] = self.peak[second_root]
            self.peak_index[first_root] = self.peak_index[second_root]
        return first_root


def _neighbours(index: int, height: int, width: int) -> Iterable[int]:
    y, x = divmod(index, width)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dx or dy):
                continue
            next_y, next_x = y + dy, x + dx
            if 0 <= next_y < height and 0 <= next_x < width:
                yield next_y * width + next_x


def _meaningful_candidates(
    nodes: Sequence[EvidenceComponentNode],
    *,
    maximum_candidates: int,
    minimum_peak_ratio: float,
    minimum_area_growth: float,
    minimum_mass_growth: float,
    maximum_area_ratio: float,
) -> tuple[int, ...]:
    """Compress long unary chains without introducing fixed spatial scales."""

    global_peak = max((node.peak for node in nodes), default=0.0)
    children = {node.node_id: node.child_ids for node in nodes}
    leaves = [
        node
        for node in nodes
        if not children[node.node_id]
        and node.peak >= minimum_peak_ratio * global_peak
    ]
    kept: set[int] = set()
    for leaf in leaves:
        current = leaf
        last_kept = leaf
        if current.area_ratio <= maximum_area_ratio:
            kept.add(current.node_id)
        while current.parent_id is not None:
            parent = nodes[current.parent_id]
            area_growth = parent.area_ratio / max(last_kept.area_ratio, 1e-12) - 1.0
            mass_growth = (
                parent.component_mass / max(last_kept.component_mass, 1e-12) - 1.0
            )
            is_merge = len(parent.child_ids) > 1
            if (
                parent.area_ratio <= maximum_area_ratio
                and (
                    is_merge
                    or area_growth >= minimum_area_growth
                    or mass_growth >= minimum_mass_growth
                )
            ):
                kept.add(parent.node_id)
                last_kept = parent
            current = parent

    # Remove exact duplicate crop boxes.  Prefer the higher-threshold node,
    # which is the tighter connected support for the same deployed rectangle.
    best_by_box: dict[tuple[int, int, int, int], EvidenceComponentNode] = {}
    for node_id in kept:
        node = nodes[node_id]
        previous = best_by_box.get(node.grid_box)
        if previous is None or node.level > previous.level:
            best_by_box[node.grid_box] = node

    candidates = list(best_by_box.values())
    candidates.sort(
        key=lambda node: (
            node.component_mass / max(node.area_ratio, 1e-8) ** 0.20,
            node.peak,
            -node.area_ratio,
        ),
        reverse=True,
    )
    candidates = candidates[:maximum_candidates]
    # Stable tree order makes downstream JSON and folds reproducible.
    return tuple(sorted(node.node_id for node in candidates))


def build_evidence_component_tree(
    probability: np.ndarray,
    *,
    maximum_candidates: int = 128,
    minimum_peak_ratio: float = 0.10,
    minimum_area_growth: float = 0.20,
    minimum_mass_growth: float = 0.05,
    maximum_area_ratio: float = 0.80,
) -> EvidenceComponentTree:
    """Build an exact 8-connected max-tree from one PTEA evidence map.

    The only pruning thresholds measure *relative changes along a tree path*.
    No crop width, height, aspect ratio, or window catalogue is configured.
    """

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    if not 0 <= minimum_peak_ratio <= 1:
        raise ValueError("minimum_peak_ratio must lie in [0, 1]")
    if min(minimum_area_growth, minimum_mass_growth) < 0:
        raise ValueError("growth thresholds must be non-negative")
    if not 0 < maximum_area_ratio <= 1:
        raise ValueError("maximum_area_ratio must lie in (0, 1]")

    distribution = normalize_distribution(np.asarray(probability, dtype=np.float64))
    if distribution.ndim != 2 or distribution.size == 0:
        raise ValueError("probability must be a non-empty 2D map")
    height, width = distribution.shape
    flat = distribution.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    dsu = _DisjointSet(distribution)
    top_node = np.full(flat.size, -1, dtype=np.int64)
    mutable_nodes: list[dict[str, Any]] = []

    cursor = 0
    while cursor < len(order):
        level = float(flat[order[cursor]])
        end = cursor + 1
        while end < len(order) and float(flat[order[end]]) == level:
            end += 1
        group = [int(index) for index in order[cursor:end]]

        inherited: dict[int, set[int]] = {index: set() for index in group}
        for index in group:
            for neighbour in _neighbours(index, height, width):
                if dsu.active(neighbour):
                    node_id = int(top_node[dsu.find(neighbour)])
                    if node_id >= 0:
                        inherited[index].add(node_id)

        for index in group:
            y, x = divmod(index, width)
            dsu.activate(index, x=x, y=y)
        for index in group:
            for neighbour in _neighbours(index, height, width):
                if dsu.active(neighbour):
                    dsu.union(index, neighbour)

        children_by_root: dict[int, set[int]] = {}
        for index in group:
            root = dsu.find(index)
            children_by_root.setdefault(root, set()).update(inherited[index])

        for root, child_ids in sorted(children_by_root.items()):
            root = dsu.find(root)
            peak_index = int(dsu.peak_index[root])
            peak_y, peak_x = divmod(peak_index, width)
            node_id = len(mutable_nodes)
            node = {
                "node_id": node_id,
                "parent_id": None,
                "child_ids": tuple(sorted(child_ids)),
                "level": level,
                "peak": float(dsu.peak[root]),
                "peak_grid": (int(peak_x), int(peak_y)),
                "grid_box": (
                    int(dsu.x0[root]),
                    int(dsu.y0[root]),
                    int(dsu.x1[root]),
                    int(dsu.y1[root]),
                ),
                "cell_count": int(dsu.size[root]),
                "component_mass": float(dsu.mass[root]),
                "component_energy": float(dsu.energy[root]),
                "component_entropy_sum": float(dsu.entropy_sum[root]),
                "map_height": height,
                "map_width": width,
            }
            mutable_nodes.append(node)
            for child_id in child_ids:
                mutable_nodes[child_id]["parent_id"] = node_id
            top_node[root] = node_id
        cursor = end

    nodes = tuple(EvidenceComponentNode(**node) for node in mutable_nodes)
    roots = tuple(node.node_id for node in nodes if node.parent_id is None)
    candidate_ids = _meaningful_candidates(
        nodes,
        maximum_candidates=maximum_candidates,
        minimum_peak_ratio=minimum_peak_ratio,
        minimum_area_growth=minimum_area_growth,
        minimum_mass_growth=minimum_mass_growth,
        maximum_area_ratio=maximum_area_ratio,
    )
    return EvidenceComponentTree(
        nodes=nodes,
        root_ids=roots,
        candidate_ids=candidate_ids,
        map_height=height,
        map_width=width,
    )


def candidate_records(tree: EvidenceComponentTree) -> list[dict[str, Any]]:
    """Serialize the pruned inference candidates with parent deltas."""

    candidate_set = set(tree.candidate_ids)
    output: list[dict[str, Any]] = []
    for node in tree.candidates:
        ancestor_id = node.parent_id
        while ancestor_id is not None and ancestor_id not in candidate_set:
            ancestor_id = tree.node(ancestor_id).parent_id
        ancestor = tree.node(ancestor_id) if ancestor_id is not None else None
        output.append(
            {
                "node_id": node.node_id,
                "map_height": node.map_height,
                "map_width": node.map_width,
                "candidate_parent_id": ancestor_id,
                "child_count": len(node.child_ids),
                "bbox": node.policy_box,
                "peak_grid": list(node.peak_grid),
                "level": node.level,
                "threshold_ratio": node.threshold_ratio,
                "peak": node.peak,
                "cell_count": node.cell_count,
                "component_mass": node.component_mass,
                "component_density": node.component_density,
                "component_energy": node.component_energy,
                "component_entropy_sum": node.component_entropy_sum,
                "area_ratio": node.area_ratio,
                "fill_ratio": node.fill_ratio,
                "aspect_ratio": node.aspect_ratio,
                "parent_area_growth": (
                    ancestor.area_ratio / max(node.area_ratio, 1e-12) - 1.0
                    if ancestor is not None
                    else 0.0
                ),
                "parent_mass_growth": (
                    ancestor.component_mass / max(node.component_mass, 1e-12) - 1.0
                    if ancestor is not None
                    else 0.0
                ),
            }
        )
    return output


__all__ = [
    "EvidenceComponentNode",
    "EvidenceComponentTree",
    "build_evidence_component_tree",
    "candidate_records",
]
