"""
Persistent Priority Queue — Indexed Min-Max Heap Implementation.

A double-ended priority queue backed by a min-max heap (Atkinson, Sack,
Santoro & Strothotte, 1986) with an ``item_id → heap-index`` map for O(1)
positional lookup.  State is persisted to a JSON file using atomic writes.

All seven required operations run in O(log n) time or better:

    insert, extract_min, extract_max, peek, update, delete, is_empty

Items are identified by unique string IDs.  Priorities are numeric (int or
float — ``bool``, ``NaN``, and infinities are rejected).  Among items with
equal priority, FIFO ordering is maintained based on original insertion
order.

Persistence format
------------------
A single JSON file stores the queue version, a monotonic sequence counter,
and the list of items.  Writes follow a write-tmp → fsync → atomic-replace
strategy so that a crash mid-write never corrupts the primary file.

Zero external dependencies — uses only the Python standard library.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Union


# ---------------------------------------------------------------------------
# Internal heap entry
# ---------------------------------------------------------------------------

class _HeapEntry:
    """A single entry in the min-max heap.

    Comparison key is ``(priority, seq)`` to ensure a strict total order.
    Lower priority = higher urgency.  Among equal priorities, lower ``seq``
    (earlier insertion) comes first.
    """

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
        """Return the comparison key for heap ordering."""
        return (self.priority, self.seq)

    def to_dict(self) -> dict:
        """Return a user-facing dictionary representation."""
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


# ---------------------------------------------------------------------------
# Input validators
# ---------------------------------------------------------------------------

def _validate_priority(priority: Any) -> None:
    """Raise ``TypeError`` / ``ValueError`` if *priority* is not a finite
    ``int`` or ``float``.  Rejects ``bool``, ``NaN``, and ``±inf``.
    """
    if isinstance(priority, bool):
        raise TypeError("Priority must be an int or float, not bool")
    if not isinstance(priority, (int, float)):
        raise TypeError("Priority must be an int or float")
    if isinstance(priority, float) and (math.isnan(priority) or math.isinf(priority)):
        raise ValueError("Priority must be finite; NaN and infinity are not allowed")


def _validate_item_id(item_id: Any) -> None:
    """Raise ``TypeError`` if *item_id* is not a ``str``."""
    if not isinstance(item_id, str):
        raise TypeError("item_id must be a string")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PersistentPriorityQueue:
    """A persistent double-ended priority queue backed by a min-max heap.

    Supports O(log n) insert, extract_min, extract_max, update, and delete.
    Supports O(1) peek (minimum) and is_empty.

    State is persisted to a JSON file after every mutating operation using
    atomic writes (write-to-temp → fsync → ``os.replace``).

    Items are identified by unique **string** IDs.  Priorities are numeric
    (``int`` or ``float``; ``bool``, ``NaN``, and infinities are rejected).
    Among items with equal priority, FIFO ordering is maintained based on
    the item's **original** insertion order — ``update()`` does not change
    the tie-breaking position.

    Parameters
    ----------
    storage_path : str
        Path to the JSON persistence file.  Parent directories are created
        automatically on the first write.  Defaults to ``"data/queue.json"``.

    Raises
    ------
    ValueError
        If the persistence file exists but is malformed, has an unsupported
        schema version, contains invalid data types, or has duplicate item
        IDs.  The constructor fails fast — corrupted state is never silently
        recovered.
    """

    def __init__(self, storage_path: str = "data/queue.json") -> None:
        self._storage_path: Path = Path(storage_path)
        self._heap: list[_HeapEntry] = []
        self._index: dict[str, int] = {}  # item_id → position in _heap
        self._next_seq: int = 0  # next insertion sequence number
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
        """Insert a new item into the queue.

        Parameters
        ----------
        item_id : str
            Unique identifier for the item.
        priority : int | float
            Numeric priority (lower = higher urgency).
        value : Any, optional
            Arbitrary JSON-serialisable payload.

        Raises
        ------
        TypeError
            If *item_id* is not ``str``, or *priority* is not ``int``/``float``.
        ValueError
            If *item_id* already exists, or *priority* is ``NaN``/infinite.
        """
        _validate_item_id(item_id)
        _validate_priority(priority)
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
        """Remove and return the minimum-priority item.

        Returns
        -------
        dict
            ``{"item_id": str, "priority": int|float, "value": Any}``

        Raises
        ------
        KeyError
            If the queue is empty.
        """
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._remove_at(0)

    def extract_max(self) -> dict:
        """Remove and return the maximum-priority item.

        Returns
        -------
        dict
            ``{"item_id": str, "priority": int|float, "value": Any}``

        Raises
        ------
        KeyError
            If the queue is empty.
        """
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._remove_at(self._find_max_index())

    def peek(self) -> dict:
        """Return the **minimum-priority** item without removing it.

        This always returns the item with the lowest priority value (highest
        urgency).  To inspect the maximum, use ``extract_max()``.

        Returns
        -------
        dict
            ``{"item_id": str, "priority": int|float, "value": Any}``

        Raises
        ------
        KeyError
            If the queue is empty.
        """
        if not self._heap:
            raise KeyError("Queue is empty")
        return self._heap[0].to_dict()

    def update(self, item_id: str, new_priority: Union[int, float]) -> None:
        """Update the priority of an existing item.

        The item's original insertion sequence number is **preserved** so
        that FIFO tie-breaking among equal priorities remains based on when
        the item was first inserted.

        Parameters
        ----------
        item_id : str
            The ID of the item to update.
        new_priority : int | float
            The new priority value.

        Raises
        ------
        TypeError
            If *item_id* is not ``str``, or *new_priority* is not numeric.
        ValueError
            If *new_priority* is ``NaN`` or infinite.
        KeyError
            If *item_id* is not found in the queue.
        """
        _validate_item_id(item_id)
        _validate_priority(new_priority)
        if item_id not in self._index:
            raise KeyError(f"Item '{item_id}' not found")

        pos = self._index[item_id]
        self._heap[pos].priority = new_priority
        # Preserve original seq — do NOT assign a new sequence number.
        #
        # After updating the priority, the element may need to move up or
        # down.  _push_up may swap it with an ancestor, leaving a
        # displaced element at `pos`.  We must then push_down from `pos`
        # to fix that displaced element's relationship with its children.
        new_pos = self._push_up(pos)
        if new_pos != pos:
            # Element moved up — fix the displaced element at `pos`.
            self._push_down(pos)
        else:
            # Element stayed — it might need to move down.
            self._push_down(pos)
        self._save()

    def delete(self, item_id: str) -> dict:
        """Remove a specific item by its ID.

        Parameters
        ----------
        item_id : str
            The ID of the item to delete.

        Returns
        -------
        dict
            ``{"item_id": str, "priority": int|float, "value": Any}``
            of the removed item.

        Raises
        ------
        TypeError
            If *item_id* is not ``str``.
        KeyError
            If *item_id* is not found in the queue.
        """
        _validate_item_id(item_id)
        if item_id not in self._index:
            raise KeyError(f"Item '{item_id}' not found")
        return self._remove_at(self._index[item_id])

    def is_empty(self) -> bool:
        """Return ``True`` if the queue contains no items."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Return the number of items in the queue."""
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
        """Remove the element at heap position *pos* and return it as a dict.

        After removal, the heap property and index map are restored, and the
        new state is persisted.
        """
        entry = self._heap[pos]
        result = entry.to_dict()
        last = len(self._heap) - 1

        if pos == last:
            # Removing the last element — no heap fix needed.
            self._heap.pop()
            del self._index[entry.item_id]
        else:
            # Move the last element into the vacated position.
            self._swap(pos, last)
            self._heap.pop()
            del self._index[entry.item_id]
            # Restore heap order for the element now at *pos*.
            # _push_up may move it upward; if so, fix *pos* afterward.
            new_pos = self._push_up(pos)
            if new_pos != pos:
                self._push_down(pos)
            else:
                self._push_down(pos)

        self._save()
        return result

    def _find_max_index(self) -> int:
        """Return the heap index of the maximum element.

        In a min-max heap the maximum is the root when ``size == 1``, or
        the larger of the root's two children (indices 1 and 2).
        """
        n = len(self._heap)
        if n <= 1:
            return 0
        if n == 2:
            return 1
        # The maximum lives among the root's children on max-level 1.
        return 1 if self._heap[1]._key() >= self._heap[2]._key() else 2

    # ------------------------------------------------------------------
    # Internal: min-max heap operations
    # ------------------------------------------------------------------

    @staticmethod
    def _is_min_level(i: int) -> bool:
        """Return ``True`` if index *i* is on a min level (even depth).

        Level = floor(log₂(i + 1)).  Using ``bit_length()`` avoids any
        floating-point issues.
        """
        level = (i + 1).bit_length() - 1
        return level % 2 == 0

    def _swap(self, i: int, j: int) -> None:
        """Swap two heap entries and update the index map atomically."""
        heap = self._heap
        idx = self._index
        idx[heap[i].item_id] = j
        idx[heap[j].item_id] = i
        heap[i], heap[j] = heap[j], heap[i]

    # -- push up -------------------------------------------------------

    def _push_up(self, i: int) -> int:
        """Push element at index *i* upward.  Returns its final position."""
        if i == 0:
            return 0

        parent = (i - 1) // 2

        if self._is_min_level(i):
            # On a min level.
            if self._heap[i]._key() > self._heap[parent]._key():
                self._swap(i, parent)
                return self._push_up_max(parent)
            return self._push_up_min(i)
        else:
            # On a max level.
            if self._heap[i]._key() < self._heap[parent]._key():
                self._swap(i, parent)
                return self._push_up_min(parent)
            return self._push_up_max(i)

    def _push_up_min(self, i: int) -> int:
        """Bubble *i* upward through min (even) levels via grandparents."""
        while i >= 3:
            grandparent = ((i - 1) // 2 - 1) // 2
            if self._heap[i]._key() < self._heap[grandparent]._key():
                self._swap(i, grandparent)
                i = grandparent
            else:
                break
        return i

    def _push_up_max(self, i: int) -> int:
        """Bubble *i* upward through max (odd) levels via grandparents."""
        while i >= 3:
            grandparent = ((i - 1) // 2 - 1) // 2
            if self._heap[i]._key() > self._heap[grandparent]._key():
                self._swap(i, grandparent)
                i = grandparent
            else:
                break
        return i

    # -- push down -----------------------------------------------------

    def _push_down(self, i: int) -> None:
        """Push element at index *i* downward to restore heap order."""
        if self._is_min_level(i):
            self._push_down_min(i)
        else:
            self._push_down_max(i)

    def _push_down_min(self, i: int) -> None:
        """Trickle-down on a min level (standard min-max heap algorithm)."""
        while True:
            m, is_grandchild = self._smallest_descendant(i)
            if m is None:
                return  # Leaf node.

            if is_grandchild:
                if self._heap[m]._key() < self._heap[i]._key():
                    self._swap(m, i)
                    # The element now at *m* may violate the max property
                    # with its parent (which is on a max level).
                    parent_m = (m - 1) // 2
                    if self._heap[m]._key() > self._heap[parent_m]._key():
                        self._swap(m, parent_m)
                    i = m  # Continue pushing down from grandchild position.
                else:
                    return
            else:
                # *m* is a direct child — swap if needed and stop.
                if self._heap[m]._key() < self._heap[i]._key():
                    self._swap(m, i)
                return

    def _push_down_max(self, i: int) -> None:
        """Trickle-down on a max level (symmetric to ``_push_down_min``)."""
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

    # -- descendant finders --------------------------------------------

    def _smallest_descendant(self, i: int) -> tuple:
        """Return ``(index, is_grandchild)`` of the smallest child or
        grandchild, or ``(None, False)`` if *i* is a leaf.
        """
        n = len(self._heap)
        best = None
        best_key = None
        is_gc = False

        # Children: indices 2i+1, 2i+2
        for c in range(2 * i + 1, min(2 * i + 3, n)):
            key = self._heap[c]._key()
            if best is None or key < best_key:
                best, best_key, is_gc = c, key, False

        # Grandchildren: indices 4i+3 .. 4i+6
        for gc in range(4 * i + 3, min(4 * i + 7, n)):
            key = self._heap[gc]._key()
            if best is None or key < best_key:
                best, best_key, is_gc = gc, key, True

        return best, is_gc

    def _largest_descendant(self, i: int) -> tuple:
        """Return ``(index, is_grandchild)`` of the largest child or
        grandchild, or ``(None, False)`` if *i* is a leaf.
        """
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

    # -- heapify -------------------------------------------------------

    def _heapify(self) -> None:
        """Build a valid min-max heap from ``self._heap`` in O(n) time.

        Uses Floyd's bottom-up algorithm: call ``_push_down`` from the last
        internal node down to the root.  Also rebuilds ``self._index``.
        """
        n = len(self._heap)
        for i in range(n // 2 - 1, -1, -1):
            self._push_down(i)
        self._index = {
            entry.item_id: pos for pos, entry in enumerate(self._heap)
        }

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist the current queue state to disk.

        Strategy: write to a temporary file, ``fsync``, then atomically
        replace the original.  A crash before ``os.replace`` leaves the
        previous valid file intact.
        """
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

        tmp_path = self._storage_path.with_name(
            self._storage_path.name + ".tmp"
        )

        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, self._storage_path)

    def _load(self) -> None:
        """Load queue state from disk, or start with an empty queue.

        Raises
        ------
        ValueError
            If the file exists but is malformed, has an unsupported version,
            contains invalid data types, or has duplicate item IDs.  No
            automatic recovery is attempted — fail fast.
        """
        if not self._storage_path.exists():
            return  # Start with an empty queue.

        # -- Parse JSON ------------------------------------------------
        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Corrupted persistence file '{self._storage_path}': "
                f"invalid JSON"
            ) from exc

        # -- Validate root structure -----------------------------------
        if not isinstance(data, dict):
            raise ValueError(
                "Persistence file: root must be a JSON object"
            )
        if data.get("version") != 1:
            raise ValueError(
                f"Persistence file: unsupported version "
                f"{data.get('version')!r}"
            )

        next_seq = data.get("next_seq")
        if (
            not isinstance(next_seq, int)
            or isinstance(next_seq, bool)
            or next_seq < 0
        ):
            raise ValueError(
                "Persistence file: 'next_seq' must be a non-negative integer"
            )

        items_raw = data.get("items")
        if not isinstance(items_raw, list):
            raise ValueError("Persistence file: 'items' must be a list")

        # -- Validate and build entries --------------------------------
        seen_ids: set = set()
        entries: list[_HeapEntry] = []

        for idx, item in enumerate(items_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Persistence file: item at index {idx} must be a "
                    f"JSON object"
                )

            item_id = item.get("item_id")
            if not isinstance(item_id, str):
                raise ValueError(
                    f"Persistence file: item_id at index {idx} must be a "
                    f"string, got {type(item_id).__name__}"
                )
            if item_id in seen_ids:
                raise ValueError(
                    f"Persistence file: duplicate item_id '{item_id}'"
                )
            seen_ids.add(item_id)

            priority = item.get("priority")
            if isinstance(priority, bool) or not isinstance(
                priority, (int, float)
            ):
                raise ValueError(
                    f"Persistence file: priority for '{item_id}' must be "
                    f"int or float"
                )
            if isinstance(priority, float) and (
                math.isnan(priority) or math.isinf(priority)
            ):
                raise ValueError(
                    f"Persistence file: priority for '{item_id}' must be "
                    f"finite"
                )

            seq = item.get("seq")
            if not isinstance(seq, int) or isinstance(seq, bool):
                raise ValueError(
                    f"Persistence file: seq for '{item_id}' must be an "
                    f"integer"
                )

            entries.append(
                _HeapEntry(priority, seq, item_id, item.get("value"))
            )

        self._heap = entries
        self._next_seq = next_seq
        # Build a valid min-max heap from the loaded entries (O(n)).
        self._heapify()

    # ------------------------------------------------------------------
    # Internal: invariant validation (for tests / debugging)
    # ------------------------------------------------------------------

    def _validate_invariants(self) -> None:
        """Verify all internal invariants.

        Checks
        ------
        1. Index-map size equals heap size.
        2. Every index-map entry points to the correct heap position.
        3. The min-max heap ordering property holds for every node
           (checked against children *and* grandchildren).

        Raises
        ------
        AssertionError
            On any invariant violation, with a descriptive message.
        """
        n = len(self._heap)

        # 1. Size consistency.
        assert len(self._index) == n, (
            f"Index-map size {len(self._index)} != heap size {n}"
        )

        # 2. Index-map consistency.
        for i, entry in enumerate(self._heap):
            assert entry.item_id in self._index, (
                f"item_id '{entry.item_id}' at heap[{i}] missing from "
                f"index map"
            )
            assert self._index[entry.item_id] == i, (
                f"Index map says '{entry.item_id}' is at "
                f"{self._index[entry.item_id]}, but it is at {i}"
            )

        # 3. Min-max heap property.
        for i in range(n):
            key_i = self._heap[i]._key()
            is_min = self._is_min_level(i)

            # Check children.
            for c in range(2 * i + 1, min(2 * i + 3, n)):
                key_c = self._heap[c]._key()
                if is_min:
                    assert key_i <= key_c, (
                        f"Min-level violation: heap[{i}]={key_i} > "
                        f"child heap[{c}]={key_c}"
                    )
                else:
                    assert key_i >= key_c, (
                        f"Max-level violation: heap[{i}]={key_i} < "
                        f"child heap[{c}]={key_c}"
                    )

            # Check grandchildren.
            for gc in range(4 * i + 3, min(4 * i + 7, n)):
                key_gc = self._heap[gc]._key()
                if is_min:
                    assert key_i <= key_gc, (
                        f"Min-level violation: heap[{i}]={key_i} > "
                        f"grandchild heap[{gc}]={key_gc}"
                    )
                else:
                    assert key_i >= key_gc, (
                        f"Max-level violation: heap[{i}]={key_i} < "
                        f"grandchild heap[{gc}]={key_gc}"
                    )
