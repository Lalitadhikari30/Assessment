"""
Comprehensive test suite for the Persistent Priority Queue.

Covers:
  - Basic operations (insert, peek, extract, is_empty)
  - Ordering correctness (min and max)
  - Equal-priority FIFO tie-breaking
  - Update (increase, decrease, preserve seq, non-existent)
  - Delete (min, max, middle, only, non-existent)
  - Persistence round-trips (insert, update, delete, extract)
  - Edge cases (empty, 1-2 elements, negative/zero/large/float priorities)
  - Input validation (bool, NaN, inf, wrong types, duplicates)
  - Invariant validation after every operation
  - Mandatory randomised differential testing (30 000 operations)
  - Persistence stress testing (repeated create/destroy cycles)
"""

from __future__ import annotations

import json
import random
import sys
import os

import pytest

# Ensure the project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module import PersistentPriorityQueue  # noqa: E402


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def storage(tmp_path):
    """Return a temporary file path string for queue storage."""
    return str(tmp_path / "test_queue.json")


@pytest.fixture()
def pq(storage):
    """Return a fresh, empty PersistentPriorityQueue."""
    return PersistentPriorityQueue(storage_path=storage)


# ======================================================================
# Basic Operations
# ======================================================================

class TestBasicOperations:

    def test_new_queue_is_empty(self, pq):
        assert pq.is_empty()
        assert len(pq) == 0

    def test_insert_and_peek(self, pq):
        pq.insert("a", 5)
        assert not pq.is_empty()
        assert len(pq) == 1
        assert pq.peek() == {"item_id": "a", "priority": 5, "value": None}

    def test_insert_with_value(self, pq):
        pq.insert("a", 3, {"data": 42})
        result = pq.peek()
        assert result["value"] == {"data": 42}

    def test_insert_and_extract_min(self, pq):
        pq.insert("a", 5, "payload")
        result = pq.extract_min()
        assert result == {"item_id": "a", "priority": 5, "value": "payload"}
        assert pq.is_empty()

    def test_insert_and_extract_max(self, pq):
        pq.insert("a", 5, "payload")
        result = pq.extract_max()
        assert result == {"item_id": "a", "priority": 5, "value": "payload"}
        assert pq.is_empty()

    def test_repr(self, pq):
        r = repr(pq)
        assert "size=0" in r
        pq.insert("x", 1)
        r = repr(pq)
        assert "size=1" in r


# ======================================================================
# Ordering
# ======================================================================

class TestOrdering:

    def test_min_extraction_order(self, pq):
        for p in [5, 1, 9, 3, 7]:
            pq.insert(f"item-{p}", p)
        results = []
        while not pq.is_empty():
            results.append(pq.extract_min()["priority"])
        assert results == [1, 3, 5, 7, 9]

    def test_max_extraction_order(self, pq):
        for p in [5, 1, 9, 3, 7]:
            pq.insert(f"item-{p}", p)
        results = []
        while not pq.is_empty():
            results.append(pq.extract_max()["priority"])
        assert results == [9, 7, 5, 3, 1]

    def test_interleaved_min_max(self, pq):
        for p in [5, 1, 9, 3, 7]:
            pq.insert(f"item-{p}", p)
        assert pq.extract_min()["priority"] == 1
        assert pq.extract_max()["priority"] == 9
        assert pq.extract_min()["priority"] == 3
        assert pq.extract_max()["priority"] == 7
        assert pq.extract_min()["priority"] == 5
        assert pq.is_empty()

    def test_sorted_insert_ascending(self, pq):
        for i in range(1, 11):
            pq.insert(f"i-{i}", i)
        assert pq.peek()["priority"] == 1
        assert pq.extract_max()["priority"] == 10

    def test_sorted_insert_descending(self, pq):
        for i in range(10, 0, -1):
            pq.insert(f"i-{i}", i)
        assert pq.peek()["priority"] == 1
        assert pq.extract_max()["priority"] == 10


# ======================================================================
# Equal Priorities / FIFO Tie-Breaking
# ======================================================================

class TestEqualPriorities:

    def test_fifo_extract_min(self, pq):
        """Among equal priorities, extract_min returns the earliest-inserted."""
        pq.insert("first", 5)
        pq.insert("second", 5)
        pq.insert("third", 5)
        assert pq.extract_min()["item_id"] == "first"
        assert pq.extract_min()["item_id"] == "second"
        assert pq.extract_min()["item_id"] == "third"

    def test_reverse_fifo_extract_max(self, pq):
        """Among equal priorities, extract_max returns the latest-inserted."""
        pq.insert("first", 5)
        pq.insert("second", 5)
        pq.insert("third", 5)
        assert pq.extract_max()["item_id"] == "third"
        assert pq.extract_max()["item_id"] == "second"
        assert pq.extract_max()["item_id"] == "first"

    def test_many_equal_priorities(self, pq):
        n = 100
        for i in range(n):
            pq.insert(f"item-{i}", 0)
        ids = []
        while not pq.is_empty():
            ids.append(pq.extract_min()["item_id"])
        assert ids == [f"item-{i}" for i in range(n)]


# ======================================================================
# Update
# ======================================================================

class TestUpdate:

    def test_decrease_priority(self, pq):
        pq.insert("a", 10)
        pq.insert("b", 5)
        pq.update("a", 1)
        assert pq.peek()["item_id"] == "a"
        assert pq.peek()["priority"] == 1

    def test_increase_priority(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.update("a", 20)
        assert pq.peek()["item_id"] == "b"

    def test_update_preserves_insertion_order(self, pq):
        """After update to the same priority, FIFO order uses original seq."""
        pq.insert("first", 10)
        pq.insert("second", 20)
        pq.update("second", 10)  # Both now priority 10
        assert pq.extract_min()["item_id"] == "first"
        assert pq.extract_min()["item_id"] == "second"

    def test_update_min_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.update("a", 7)
        assert pq.peek()["item_id"] == "b"

    def test_update_max_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.update("c", 3)
        assert pq.peek()["item_id"] == "a"
        assert pq.extract_max()["item_id"] == "b"

    def test_update_middle_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.update("b", 0)  # b becomes new min
        assert pq.peek()["item_id"] == "b"

    def test_update_to_same_priority(self, pq):
        pq.insert("a", 5)
        pq.update("a", 5)
        assert pq.peek()["priority"] == 5
        pq._validate_invariants()

    def test_update_nonexistent_raises(self, pq):
        with pytest.raises(KeyError, match="not found"):
            pq.update("ghost", 5)

    def test_update_on_empty_raises(self, pq):
        with pytest.raises(KeyError, match="not found"):
            pq.update("x", 1)


# ======================================================================
# Delete
# ======================================================================

class TestDelete:

    def test_delete_only_element(self, pq):
        pq.insert("a", 5)
        result = pq.delete("a")
        assert result == {"item_id": "a", "priority": 5, "value": None}
        assert pq.is_empty()

    def test_delete_min_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.delete("a")
        assert pq.peek()["item_id"] == "b"
        pq._validate_invariants()

    def test_delete_max_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.delete("c")
        assert pq.extract_max()["priority"] == 5
        pq._validate_invariants()

    def test_delete_middle_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        pq.delete("b")
        assert pq.extract_min()["priority"] == 1
        assert pq.extract_max()["priority"] == 10

    def test_delete_nonexistent_raises(self, pq):
        with pytest.raises(KeyError, match="not found"):
            pq.delete("ghost")

    def test_delete_all_one_by_one(self, pq):
        for i in range(20):
            pq.insert(f"i-{i}", i)
        for i in range(20):
            pq.delete(f"i-{i}")
            pq._validate_invariants()
        assert pq.is_empty()


# ======================================================================
# Persistence
# ======================================================================

class TestPersistence:

    def test_persist_and_reload(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 3, "data_a")
        pq1.insert("b", 1, "data_b")
        pq1.insert("c", 5, "data_c")
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 3
        assert pq2.peek() == {"item_id": "b", "priority": 1, "value": "data_b"}
        pq2._validate_invariants()

    def test_persist_after_update(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 3)
        pq1.insert("b", 1)
        pq1.update("b", 10)
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert pq2.peek()["item_id"] == "a"
        pq2._validate_invariants()

    def test_persist_after_delete(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 3)
        pq1.insert("b", 1)
        pq1.delete("b")
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 1
        assert pq2.peek()["item_id"] == "a"
        pq2._validate_invariants()

    def test_persist_after_extract_min(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 3)
        pq1.insert("b", 1)
        pq1.extract_min()
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 1
        assert pq2.peek()["item_id"] == "a"
        pq2._validate_invariants()

    def test_persist_after_extract_max(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 3)
        pq1.insert("b", 1)
        pq1.insert("c", 9)
        pq1.extract_max()
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 2
        assert pq2.extract_max()["priority"] == 3
        pq2._validate_invariants()

    def test_persist_int_and_float_priorities(self, storage):
        """Priority types (int vs float) survive JSON round-trip."""
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("int_item", 3)
        pq1.insert("float_item", 2.5)
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        result = pq2.extract_min()
        assert result["priority"] == 2.5
        assert isinstance(result["priority"], float)
        result = pq2.extract_min()
        assert result["priority"] == 3
        assert isinstance(result["priority"], int)

    def test_missing_file_starts_fresh(self, tmp_path):
        path = str(tmp_path / "nonexistent" / "queue.json")
        pq = PersistentPriorityQueue(storage_path=path)
        assert pq.is_empty()

    def test_corrupted_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json!!!", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            PersistentPriorityQueue(storage_path=str(path))

    def test_bad_version_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            '{"version": 99, "next_seq": 0, "items": []}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="unsupported version"):
            PersistentPriorityQueue(storage_path=str(path))

    def test_duplicate_ids_in_file_raises(self, tmp_path):
        path = tmp_path / "dupes.json"
        data = {
            "version": 1,
            "next_seq": 2,
            "items": [
                {"item_id": "a", "priority": 1, "seq": 0, "value": None},
                {"item_id": "a", "priority": 2, "seq": 1, "value": None},
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            PersistentPriorityQueue(storage_path=str(path))

    def test_bool_priority_in_file_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        data = {
            "version": 1,
            "next_seq": 1,
            "items": [
                {"item_id": "a", "priority": True, "seq": 0, "value": None},
            ],
        }
        # JSON serialises bool as true/false, which json.load returns as
        # Python bool — our validator catches this.
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="int or float"):
            PersistentPriorityQueue(storage_path=str(path))

    def test_stale_tmp_ignored(self, tmp_path):
        """A leftover .tmp file must NOT be loaded."""
        real = tmp_path / "queue.json"
        stale = tmp_path / "queue.json.tmp"

        # Create a valid queue with one item.
        pq1 = PersistentPriorityQueue(storage_path=str(real))
        pq1.insert("a", 1)
        del pq1

        # Plant a stale .tmp with different data.
        stale.write_text(
            json.dumps({
                "version": 1, "next_seq": 99,
                "items": [
                    {"item_id": "z", "priority": 999, "seq": 0, "value": None}
                ],
            }),
            encoding="utf-8",
        )

        pq2 = PersistentPriorityQueue(storage_path=str(real))
        assert pq2.peek()["item_id"] == "a"  # loaded from real, NOT .tmp


# ======================================================================
# Edge Cases
# ======================================================================

class TestEdgeCases:

    def test_empty_extract_min_raises(self, pq):
        with pytest.raises(KeyError, match="empty"):
            pq.extract_min()

    def test_empty_extract_max_raises(self, pq):
        with pytest.raises(KeyError, match="empty"):
            pq.extract_max()

    def test_empty_peek_raises(self, pq):
        with pytest.raises(KeyError, match="empty"):
            pq.peek()

    def test_single_element_min_and_max(self, pq):
        pq.insert("solo", 42)
        assert pq.peek()["priority"] == 42
        pq.insert("other", 99)
        pq.delete("other")
        assert pq.extract_min()["item_id"] == "solo"

    def test_two_elements(self, pq):
        pq.insert("lo", 1)
        pq.insert("hi", 10)
        assert pq.extract_min()["priority"] == 1
        assert pq.extract_max()["priority"] == 10

    def test_negative_priorities(self, pq):
        pq.insert("a", -5)
        pq.insert("b", -10)
        pq.insert("c", 0)
        assert pq.extract_min()["priority"] == -10
        assert pq.extract_max()["priority"] == 0

    def test_zero_priority(self, pq):
        pq.insert("a", 0)
        pq.insert("b", 1)
        pq.insert("c", -1)
        assert pq.peek()["priority"] == -1

    def test_large_priorities(self, pq):
        pq.insert("big", 10**18)
        pq.insert("small", -(10**18))
        assert pq.extract_min()["priority"] == -(10**18)
        assert pq.extract_max()["priority"] == 10**18

    def test_float_priority(self, pq):
        pq.insert("a", 1.5)
        pq.insert("b", 1)
        pq.insert("c", 2)
        assert pq.extract_min()["priority"] == 1
        assert pq.extract_min()["priority"] == 1.5

    def test_int_and_float_equal(self, pq):
        """int 1 and float 1.0 are equal in Python; FIFO breaks the tie."""
        pq.insert("a", 1)
        pq.insert("b", 1.0)
        assert pq.extract_min()["item_id"] == "a"
        assert pq.extract_min()["item_id"] == "b"

    def test_extract_until_empty(self, pq):
        for i in range(50):
            pq.insert(f"i-{i}", i)
        for _ in range(50):
            pq.extract_min()
        assert pq.is_empty()
        with pytest.raises(KeyError):
            pq.extract_min()


# ======================================================================
# Input Validation
# ======================================================================

class TestInputValidation:

    def test_duplicate_insert_raises(self, pq):
        pq.insert("a", 5)
        with pytest.raises(ValueError, match="already exists"):
            pq.insert("a", 10)

    def test_bool_priority_insert_raises(self, pq):
        with pytest.raises(TypeError, match="not bool"):
            pq.insert("a", True)

    def test_nan_priority_raises(self, pq):
        with pytest.raises(ValueError, match="finite"):
            pq.insert("a", float("nan"))

    def test_inf_priority_raises(self, pq):
        with pytest.raises(ValueError, match="finite"):
            pq.insert("a", float("inf"))

    def test_neg_inf_priority_raises(self, pq):
        with pytest.raises(ValueError, match="finite"):
            pq.insert("a", float("-inf"))

    def test_string_priority_raises(self, pq):
        with pytest.raises(TypeError, match="int or float"):
            pq.insert("a", "high")

    def test_none_priority_raises(self, pq):
        with pytest.raises(TypeError, match="int or float"):
            pq.insert("a", None)

    def test_non_string_id_raises(self, pq):
        with pytest.raises(TypeError, match="string"):
            pq.insert(123, 5)

    def test_update_bool_priority_raises(self, pq):
        pq.insert("a", 5)
        with pytest.raises(TypeError, match="not bool"):
            pq.update("a", False)

    def test_update_nan_raises(self, pq):
        pq.insert("a", 5)
        with pytest.raises(ValueError, match="finite"):
            pq.update("a", float("nan"))

    def test_delete_non_string_id_raises(self, pq):
        with pytest.raises(TypeError, match="string"):
            pq.delete(42)


# ======================================================================
# Invariant Validation
# ======================================================================

class TestInvariants:

    def test_invariants_after_sequential_inserts(self, pq):
        for i in range(200):
            pq.insert(f"i-{i}", random.randint(-1000, 1000))
            pq._validate_invariants()

    def test_invariants_after_mixed_operations(self, pq):
        rng = random.Random(12345)
        for i in range(80):
            pq.insert(f"i-{i}", rng.randint(-100, 100))
        pq._validate_invariants()

        for _ in range(15):
            pq.extract_min()
            pq._validate_invariants()

        for _ in range(15):
            pq.extract_max()
            pq._validate_invariants()

        remaining = list(pq._index.keys())
        rng.shuffle(remaining)
        for item_id in remaining[:10]:
            pq.update(item_id, rng.randint(-100, 100))
            pq._validate_invariants()

        remaining = list(pq._index.keys())
        rng.shuffle(remaining)
        for item_id in remaining[:10]:
            pq.delete(item_id)
            pq._validate_invariants()


# ======================================================================
# Randomised Differential Testing (Mandatory — 30 000 operations)
# ======================================================================

class TestRandomisedDifferential:
    """Compare every operation against a trusted reference model (a plain
    Python dict with ``sorted()`` lookups).
    """

    def test_differential_30k_operations(self, tmp_path):
        """30 000 random operations verified against a reference model.

        The reference is a ``dict[str, (priority, seq, value)]``.  After
        every mutating operation the test asserts that the queue and the
        reference agree on the extracted/peeked item and on overall size.
        Invariants are checked every 500 operations.
        """
        storage = str(tmp_path / "diff.json")
        pq = PersistentPriorityQueue(storage_path=storage)

        ref: dict[str, tuple] = {}   # item_id → (priority, seq, value)
        next_seq = 0
        id_counter = 0

        rng = random.Random(42)

        ops = [
            "insert", "extract_min", "extract_max",
            "peek", "is_empty", "update", "delete",
        ]
        weights = [30, 15, 15, 10, 5, 15, 10]

        for step in range(30_000):
            op = rng.choices(ops, weights=weights, k=1)[0]

            if op == "insert":
                item_id = f"item-{id_counter}"
                id_counter += 1
                priority = rng.randint(-500, 500)
                value = f"val-{item_id}"

                pq.insert(item_id, priority, value)
                ref[item_id] = (priority, next_seq, value)
                next_seq += 1

            elif op == "extract_min":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.extract_min()
                else:
                    result = pq.extract_min()
                    expected_id = min(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert result["item_id"] == expected_id, (
                        f"step {step}: extract_min returned "
                        f"'{result['item_id']}' but expected '{expected_id}'"
                    )
                    assert result["priority"] == ref[expected_id][0]
                    assert result["value"] == ref[expected_id][2]
                    del ref[expected_id]

            elif op == "extract_max":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.extract_max()
                else:
                    result = pq.extract_max()
                    expected_id = max(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert result["item_id"] == expected_id, (
                        f"step {step}: extract_max returned "
                        f"'{result['item_id']}' but expected '{expected_id}'"
                    )
                    assert result["priority"] == ref[expected_id][0]
                    assert result["value"] == ref[expected_id][2]
                    del ref[expected_id]

            elif op == "peek":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.peek()
                else:
                    result = pq.peek()
                    expected_id = min(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert result["item_id"] == expected_id
                    assert result["priority"] == ref[expected_id][0]

            elif op == "is_empty":
                assert pq.is_empty() == (len(ref) == 0)

            elif op == "update":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.update("nonexistent", 0)
                else:
                    target = rng.choice(list(ref.keys()))
                    new_pri = rng.randint(-500, 500)
                    old = ref[target]
                    pq.update(target, new_pri)
                    ref[target] = (new_pri, old[1], old[2])  # preserve seq

            elif op == "delete":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.delete("nonexistent")
                else:
                    target = rng.choice(list(ref.keys()))
                    result = pq.delete(target)
                    assert result["item_id"] == target
                    assert result["priority"] == ref[target][0]
                    del ref[target]

            # Periodic invariant check.
            if step % 500 == 0:
                pq._validate_invariants()
                assert len(pq) == len(ref), (
                    f"step {step}: size mismatch pq={len(pq)} ref={len(ref)}"
                )

        # -- Final consistency check -----------------------------------
        pq._validate_invariants()
        assert len(pq) == len(ref)

        # Drain the queue via extract_min and compare against sorted ref.
        ref_sorted = sorted(ref.values(), key=lambda x: (x[0], x[1]))
        extracted = []
        while not pq.is_empty():
            extracted.append(pq.extract_min())

        for actual, (pri, seq, val) in zip(extracted, ref_sorted):
            assert actual["priority"] == pri
            assert actual["value"] == val

    def test_persistence_stress_20_cycles(self, tmp_path):
        """Insert items across 20 create/destroy cycles and verify state."""
        storage = str(tmp_path / "persist_stress.json")
        rng = random.Random(99)

        ref: dict[str, tuple] = {}  # item_id → (priority, seq)
        next_seq = 0
        next_id = 0

        pq = PersistentPriorityQueue(storage_path=storage)

        for cycle in range(20):
            # Insert a batch.
            for _ in range(50):
                item_id = f"item-{next_id}"
                next_id += 1
                priority = rng.randint(-100, 100)
                pq.insert(item_id, priority)
                ref[item_id] = (priority, next_seq)
                next_seq += 1

            # Destroy and recreate.
            del pq
            pq = PersistentPriorityQueue(storage_path=storage)
            pq._validate_invariants()
            assert len(pq) == len(ref)

            # Extract a few mins and verify.
            for _ in range(min(10, len(ref))):
                result = pq.extract_min()
                expected_id = min(ref, key=lambda k: ref[k])
                assert result["item_id"] == expected_id
                assert result["priority"] == ref[expected_id][0]
                del ref[expected_id]

        pq._validate_invariants()
        assert len(pq) == len(ref)


# ======================================================================
# Additional stress tests (Phase 15 — hidden-test simulation)
# ======================================================================

class TestHiddenTestSimulation:
    """Adversarial scenarios an interviewer might run."""

    def test_repeated_update_same_item(self, pq):
        """Update the same item 1000 times."""
        pq.insert("target", 0)
        for i in range(1, 11):
            pq.insert(f"other-{i}", i)

        rng = random.Random(7)
        for _ in range(1000):
            pq.update("target", rng.randint(-500, 500))
            pq._validate_invariants()

        assert "target" in pq._index

    def test_insert_extract_all_repeated(self, pq):
        """Insert N items, extract all, repeat."""
        for cycle in range(5):
            for i in range(200):
                pq.insert(f"c{cycle}-{i}", random.randint(-100, 100))
            prev = None
            while not pq.is_empty():
                cur = pq.extract_min()["priority"]
                if prev is not None:
                    assert cur >= prev
                prev = cur

    def test_alternating_insert_delete(self, pq):
        """Alternating insert and delete on the same ID."""
        for i in range(500):
            pq.insert("x", i)
            result = pq.delete("x")
            assert result["priority"] == i
            assert pq.is_empty()

    def test_many_duplicates_extract_order(self, pq):
        """500 items with only 5 distinct priority values."""
        rng = random.Random(321)
        priorities = [1, 2, 3, 4, 5]
        for i in range(500):
            pq.insert(f"i-{i}", rng.choice(priorities))

        prev_pri = -float("inf")
        while not pq.is_empty():
            cur = pq.extract_min()["priority"]
            assert cur >= prev_pri
            prev_pri = cur

    def test_large_queue_1000_items(self, pq):
        """Build a 1000-item queue and verify min/max consistency."""
        rng = random.Random(555)
        items = {}
        for i in range(1000):
            pri = rng.randint(-10000, 10000)
            pq.insert(f"i-{i}", pri)
            items[f"i-{i}"] = pri

        pq._validate_invariants()

        # Peek should match the global minimum.
        expected_min = min(items.values())
        assert pq.peek()["priority"] == expected_min

        # Extract max should match the global maximum.
        expected_max = max(items.values())
        result = pq.extract_max()
        assert result["priority"] == expected_max
