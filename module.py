"""
Persistent Priority Queue backed by an Indexed Min-Max Heap.

Implements a double-ended priority queue with O(1) ID lookups and crash-resilient
JSON persistence via atomic writes (write-tmp -> fsync -> atomic replace).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Union


class _HeapEntry:
    """Entry stored in the heap with comparison key (priority, seq)."""

    __slots__ = ("priority", "seq", "item_id", "value")

    def __init__(
        self,
        priority: Union[int, float],
        seq: int,
        item_id: str,
        value: Any,
    ) -> None:
        self.priority = priority
        self.seq = seq
        self.item_id = item_id
        self.value = value

    def _key(self) -> tuple:
        """Return (priority, seq) tuple for strict deterministic ordering."""
        return (self.priority, self.seq)

    def to_dict(self) -> dict:
        """Return user-facing dictionary representation."""
        return {
            "item_id": self.item_id,
            "priority": self.priority,
            "value": self.value,
        }

    def __repr__(self) -> str:
        return (
            f"_HeapEntry(priority={self.priority!r}, seq={self.seq}, "
            f"item_id={self.item_id!r}, value={self.value!r})"
        )


def _validate_priority(priority: Any) -> None:
    """Validate that priority is a finite int or float."""
    if isinstance(priority, bool) or not isinstance(priority, (int, float)):
        raise TypeError("Priority must be an int or float")
    if isinstance(priority, float) and (math.isnan(priority) or math.isinf(priority)):
        raise ValueError("Priority must be finite; NaN and infinity are not allowed")


def _validate_item_id(item_id: Any) -> None:
    """Validate that item_id is a string."""
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a string")


def _validate_value(value: Any) -> None:
    """Ensure value is JSON-serializable before modifying in-memory queue state."""
    if value is not None:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Value must be JSON-serializable: {exc}") from exc


class PersistentPriorityQueue:
    """A double-ended persistent priority queue backed by an Indexed Min-Max Heap.

    In-Memory Complexity:
      - insert:      O(log n)
      - extract_min: O(log n)
      - extract_max: O(log n)
      - peek:        O(1)
      - update:      O(log n)
      - delete:      O(log n)
      - is_empty:    O(1)

    Persistence Overhead:
      Each mutating operation serializes queue state to JSON in O(n) time.
    """

    def __init__(self, storage_path: str = "data/queue.json") -> None:
        self._storage_path: Path = Path(storage_path)
        self._heap: list[_HeapEntry] = []
        self._index: dict[str, int] = {}
        self._next_seq: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        item_id: str,
        priority: Union[int, float],
        value: Any = None,
    ) -> None:
        """Insert a new item into the queue."""
        _validate_item_id(item_id)
        _validate_priority(priority)
        _validate_value(value)

        if item_id in self._index:
            raise ValueError(f"Item '{item_id}' already exists")

        entry = _HeapEntry(priority, self._next_seq, item_id, value)
        self._next_seq += 1

        pos = len(self._heap)
        self._heap.append(entry)
        self._index[item_id] = pos
        self._push_up(pos)
        self._save()

    def extract_min(self) -> dict:
        """Remove and return the minimum-priority (highest urgency) item."""
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._remove_at(0)

    def extract_max(self) -> dict:
        """Remove and return the maximum-priority (lowest urgency) item."""
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._remove_at(self._find_max_index())

    def peek(self) -> dict:
        """Return the minimum-priority item without removing it."""
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._heap[0].to_dict()

    def update(self, item_id: str, new_priority: Union[int, float]) -> None:
        """Update priority of an existing item, preserving original FIFO seq."""
        _validate_item_id(item_id)
        _validate_priority(new_priority)

        if item_id not in self._index:
            raise KeyError(f"Item '{item_id}' not found")

        pos = self._index[item_id]
        self._heap[pos].priority = new_priority

        self._push_up(pos)
        self._push_down(pos)
        self._save()

    def delete(self, item_id: str) -> dict:
        """Remove a specific item by its ID."""
        _validate_item_id(item_id)
        if item_id not in self._index:
            raise KeyError(f"Item '{item_id}' not found")
        return self._remove_at(self._index[item_id])

    def is_empty(self) -> bool:
        """Return True if the queue contains no items."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return (
            f"PersistentPriorityQueue(size={len(self._heap)}, "
            f"storage='{self._storage_path}')"
        )

    # ------------------------------------------------------------------
    # Internal: removal helper
    # ------------------------------------------------------------------

    def _remove_at(self, pos: int) -> dict:
        """Remove entry at position pos, restore heap order, and persist."""
        entry = self._heap[pos]
        result = entry.to_dict()
        last = len(self._heap) - 1

        if pos == last:
            self._heap.pop()
            del self._index[entry.item_id]
        else:
            self._swap(pos, last)
            self._heap.pop()
            del self._index[entry.item_id]
            self._push_up(pos)
            self._push_down(pos)

        self._save()
        return result

    def _find_max_index(self) -> int:
        """Return index of maximum element (root if size 1, else larger child)."""
        n = len(self._heap)
        if n <= 1:
            return 0
        if n == 2:
            return 1
        return 1 if self._heap[1]._key() >= self._heap[2]._key() else 2

    # ------------------------------------------------------------------
    # Internal: Min-Max Heap algorithm (Atkinson et al., 1986)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_min_level(i: int) -> bool:
        """Return True if index i is on a min-level (even depth 0, 2, ...)."""
        level = (i + 1).bit_length() - 1
        return level % 2 == 0

    def _swap(self, i: int, j: int) -> None:
        """Swap heap elements and keep index map synchronized."""
        heap = self._heap
        idx = self._index
        idx[heap[i].item_id] = j
        idx[heap[j].item_id] = i
        heap[i], heap[j] = heap[j], heap[i]

    def _push_up(self, i: int) -> int:
        """Push element at index i upward to its proper min/max level."""
        if i == 0:
            return 0

        parent = (i - 1) // 2
        if self._is_min_level(i):
            if self._heap[i]._key() > self._heap[parent]._key():
                self._swap(i, parent)
                return self._push_up_max(parent)
            return self._push_up_min(i)
        else:
            if self._heap[i]._key() < self._heap[parent]._key():
                self._swap(i, parent)
                return self._push_up_min(parent)
            return self._push_up_max(i)

    def _push_up_min(self, i: int) -> int:
        """Sift up through grandparent min-levels."""
        while i >= 3:
            grandparent = ((i - 1) // 2 - 1) // 2
            if self._heap[i]._key() < self._heap[grandparent]._key():
                self._swap(i, grandparent)
                i = grandparent
            else:
                break
        return i

    def _push_up_max(self, i: int) -> int:
        """Sift up through grandparent max-levels."""
        while i >= 3:
            grandparent = ((i - 1) // 2 - 1) // 2
            if self._heap[i]._key() > self._heap[grandparent]._key():
                self._swap(i, grandparent)
                i = grandparent
            else:
                break
        return i

    def _push_down(self, i: int) -> None:
        """Sift down from index i according to level type."""
        if self._is_min_level(i):
            self._push_down_min(i)
        else:
            self._push_down_max(i)

    def _push_down_min(self, i: int) -> None:
        """Sift down on a min-level by comparing with children and grandchildren."""
        while True:
            m, is_grandchild = self._smallest_descendant(i)
            if m is None:
                return

            if is_grandchild:
                if self._heap[m]._key() < self._heap[i]._key():
                    self._swap(m, i)
                    parent_m = (m - 1) // 2
                    if self._heap[m]._key() > self._heap[parent_m]._key():
                        self._swap(m, parent_m)
                    i = m
                else:
                    return
            else:
                if self._heap[m]._key() < self._heap[i]._key():
                    self._swap(m, i)
                return

    def _push_down_max(self, i: int) -> None:
        """Sift down on a max-level by comparing with children and grandchildren."""
        while True:
            m, is_grandchild = self._largest_descendant(i)
            if m is None:
                return

            if is_grandchild:
                if self._heap[m]._key() > self._heap[i]._key():
                    self._swap(m, i)
                    parent_m = (m - 1) // 2
                    if self._heap[m]._key() < self._heap[parent_m]._key():
                        self._swap(m, parent_m)
                    i = m
                else:
                    return
            else:
                if self._heap[m]._key() > self._heap[i]._key():
                    self._swap(m, i)
                return

    def _smallest_descendant(self, i: int) -> tuple[int | None, bool]:
        """Find smallest child/grandchild of node i."""
        n = len(self._heap)
        best = None
        best_key = None
        is_gc = False

        for c in range(2 * i + 1, min(2 * i + 3, n)):
            key = self._heap[c]._key()
            if best is None or key < best_key:
                best, best_key, is_gc = c, key, False

        for gc in range(4 * i + 3, min(4 * i + 7, n)):
            key = self._heap[gc]._key()
            if best is None or key < best_key:
                best, best_key, is_gc = gc, key, True

        return best, is_gc

    def _largest_descendant(self, i: int) -> tuple[int | None, bool]:
        """Find largest child/grandchild of node i."""
        n = len(self._heap)
        best = None
        best_key = None
        is_gc = False

        for c in range(2 * i + 1, min(2 * i + 3, n)):
            key = self._heap[c]._key()
            if best is None or key > best_key:
                best, best_key, is_gc = c, key, False

        for gc in range(4 * i + 3, min(4 * i + 7, n)):
            key = self._heap[gc]._key()
            if best is None or key > best_key:
                best, best_key, is_gc = gc, key, True

        return best, is_gc

    def _heapify(self) -> None:
        """Rebuild min-max heap in O(n) Floyd bottom-up order."""
        # Index map must exist before _push_down calls _swap
        self._index = {entry.item_id: i for i, entry in enumerate(self._heap)}
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._push_down(i)

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Atomically persist queue state to disk."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "next_seq": self._next_seq,
            "items": [
                {
                    "item_id": e.item_id,
                    "priority": e.priority,
                    "seq": e.seq,
                    "value": e.value,
                }
                for e in self._heap
            ],
        }

        tmp_path = self._storage_path.with_name(self._storage_path.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._storage_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _load(self) -> None:
        """Load queue state from disk or start empty if file does not exist."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Corrupted persistence file: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Persistence file root must be a JSON object")
        if data.get("version") != 1:
            raise ValueError(f"Unsupported schema version: {data.get('version')!r}")

        next_seq = data.get("next_seq")
        if not isinstance(next_seq, int) or isinstance(next_seq, bool) or next_seq < 0:
            raise ValueError("Persistence file 'next_seq' must be a non-negative integer")

        items_raw = data.get("items")
        if not isinstance(items_raw, list):
            raise ValueError("Persistence file 'items' must be a list")

        seen_ids: set[str] = set()
        entries: list[_HeapEntry] = []

        for idx, item in enumerate(items_raw):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {idx} must be a JSON object")

            item_id = item.get("item_id")
            if not isinstance(item_id, str):
                raise ValueError(f"item_id at index {idx} must be a string")
            if item_id in seen_ids:
                raise ValueError(f"Duplicate item_id '{item_id}' in persistence file")
            seen_ids.add(item_id)

            priority = item.get("priority")
            if isinstance(priority, bool) or not isinstance(priority, (int, float)):
                raise ValueError(f"Priority for '{item_id}' must be an int or float")
            if isinstance(priority, float) and (math.isnan(priority) or math.isinf(priority)):
                raise ValueError(f"Priority for '{item_id}' must be finite")

            seq = item.get("seq")
            if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0 or seq >= next_seq:
                raise ValueError(
                    f"seq for '{item_id}' must be an integer in range [0, {next_seq - 1}]"
                )

            entries.append(_HeapEntry(priority, seq, item_id, item.get("value")))

        self._heap = entries
        self._next_seq = next_seq
        self._heapify()

    # ------------------------------------------------------------------
    # Internal: invariant validation (used in test suite)
    # ------------------------------------------------------------------

    def _validate_invariants(self) -> None:
        """Verify internal consistency of index map and min-max heap property."""
        n = len(self._heap)
        assert len(self._index) == n, f"Index size {len(self._index)} != heap size {n}"

        for i, entry in enumerate(self._heap):
            assert entry.item_id in self._index, f"Missing item '{entry.item_id}' in index"
            assert self._index[entry.item_id] == i, f"Index mismatch for '{entry.item_id}'"

        for i in range(n):
            key_i = self._heap[i]._key()
            is_min = self._is_min_level(i)

            for c in range(2 * i + 1, min(2 * i + 3, n)):
                key_c = self._heap[c]._key()
                if is_min:
                    assert key_i <= key_c, f"Min-level child violation at {i} -> {c}"
                else:
                    assert key_i >= key_c, f"Max-level child violation at {i} -> {c}"

            for gc in range(4 * i + 3, min(4 * i + 7, n)):
                key_gc = self._heap[gc]._key()
                if is_min:
                    assert key_i <= key_gc, f"Min-level grandchild violation at {i} -> {gc}"
                else:
                    assert key_i >= key_gc, f"Max-level grandchild violation at {i} -> {gc}"
