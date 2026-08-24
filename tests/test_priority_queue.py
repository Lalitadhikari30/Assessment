"""
Test suite for the Persistent Priority Queue.

Covers:
  - Basic operations (insert, peek, extract_min, extract_max, is_empty)
  - Ordering correctness (min-order, max-order, interleaved)
  - Equal-priority FIFO tie-breaking
  - Update and delete operations
  - Persistence round-trip and reload across varied queue sizes (1..1000)
  - Input validation, malformed persistence handling, and mutation safety
  - Internal min-max heap and index-map invariant validation
  - Randomized differential testing against an independent reference oracle
"""

from __future__ import annotations

import json
import os
import random
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module import PersistentPriorityQueue


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def storage(tmp_path):
    """Return a temporary file path for queue storage."""
    return str(tmp_path / "queue.json")


@pytest.fixture()
def pq(storage):
    """Return a fresh PersistentPriorityQueue."""
    return PersistentPriorityQueue(storage_path=storage)


# ======================================================================
# Basic API
# ======================================================================

class TestBasicAPI:

    def test_new_queue_is_empty(self, pq):
        assert pq.is_empty()
        assert len(pq) == 0

    def test_insert_and_peek(self, pq):
        pq.insert("task-1", 5, {"desc": "low"})
        assert not pq.is_empty()
        assert len(pq) == 1
        assert pq.peek() == {"item_id": "task-1", "priority": 5, "value": {"desc": "low"}}

    def test_extract_min(self, pq):
        pq.insert("task-1", 5, "payload")
        assert pq.extract_min() == {"item_id": "task-1", "priority": 5, "value": "payload"}
        assert pq.is_empty()

    def test_extract_max(self, pq):
        pq.insert("task-1", 5, "payload")
        assert pq.extract_max() == {"item_id": "task-1", "priority": 5, "value": "payload"}
        assert pq.is_empty()

    def test_empty_queue_raises_key_error(self, pq):
        with pytest.raises(KeyError, match="empty"):
            pq.peek()
        with pytest.raises(KeyError, match="empty"):
            pq.extract_min()
        with pytest.raises(KeyError, match="empty"):
            pq.extract_max()


# ======================================================================
# Ordering
# ======================================================================

class TestOrdering:

    def test_min_extraction_order(self, pq):
        for p in [9, 1, 5, 3, 7]:
            pq.insert(f"item-{p}", p)
        extracted = [pq.extract_min()["priority"] for _ in range(5)]
        assert extracted == [1, 3, 5, 7, 9]

    def test_max_extraction_order(self, pq):
        for p in [9, 1, 5, 3, 7]:
            pq.insert(f"item-{p}", p)
        extracted = [pq.extract_max()["priority"] for _ in range(5)]
        assert extracted == [9, 7, 5, 3, 1]

    def test_interleaved_min_max(self, pq):
        for p in [10, 20, 30, 40, 50]:
            pq.insert(f"i-{p}", p)
        assert pq.extract_min()["priority"] == 10
        assert pq.extract_max()["priority"] == 50
        assert pq.extract_min()["priority"] == 20
        assert pq.extract_max()["priority"] == 40
        assert pq.extract_min()["priority"] == 30
        assert pq.is_empty()

    def test_ascending_and_descending_inserts(self, pq):
        for i in range(1, 21):
            pq.insert(f"asc-{i}", i)
        assert pq.peek()["priority"] == 1
        assert pq.extract_max()["priority"] == 20

        while not pq.is_empty():
            pq.extract_min()

        for i in range(20, 0, -1):
            pq.insert(f"desc-{i}", i)
        assert pq.peek()["priority"] == 1
        assert pq.extract_max()["priority"] == 20


# ======================================================================
# Equal Priorities (FIFO Tie-Breaking)
# ======================================================================

class TestEqualPriorities:

    def test_fifo_tie_breaking_min(self, pq):
        pq.insert("first", 10)
        pq.insert("second", 10)
        pq.insert("third", 10)
        assert pq.extract_min()["item_id"] == "first"
        assert pq.extract_min()["item_id"] == "second"
        assert pq.extract_min()["item_id"] == "third"

    def test_fifo_tie_breaking_max(self, pq):
        pq.insert("first", 10)
        pq.insert("second", 10)
        pq.insert("third", 10)
        assert pq.extract_max()["item_id"] == "third"
        assert pq.extract_max()["item_id"] == "second"
        assert pq.extract_max()["item_id"] == "first"

    def test_many_duplicate_priorities(self, pq):
        for i in range(100):
            pq.insert(f"item-{i}", priority=42)
        extracted_ids = [pq.extract_min()["item_id"] for _ in range(100)]
        assert extracted_ids == [f"item-{i}" for i in range(100)]


# ======================================================================
# Update Operations
# ======================================================================

class TestUpdate:

    def test_update_decrease_priority(self, pq):
        pq.insert("a", 10)
        pq.insert("b", 5)
        pq.update("a", 1)
        assert pq.peek()["item_id"] == "a"
        assert pq.peek()["priority"] == 1

    def test_update_increase_priority(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.update("a", 20)
        assert pq.peek()["item_id"] == "b"
        assert pq.extract_max()["item_id"] == "a"

    def test_update_preserves_fifo_sequence(self, pq):
        pq.insert("first", 10)
        pq.insert("second", 20)
        pq.update("second", 10)
        assert pq.extract_min()["item_id"] == "first"
        assert pq.extract_min()["item_id"] == "second"

    def test_update_nonexistent_raises(self, pq):
        with pytest.raises(KeyError, match="not found"):
            pq.update("missing", 5)


# ======================================================================
# Delete Operations
# ======================================================================

class TestDelete:

    def test_delete_arbitrary_element(self, pq):
        pq.insert("a", 1)
        pq.insert("b", 5)
        pq.insert("c", 10)
        deleted = pq.delete("b")
        assert deleted["item_id"] == "b"
        assert len(pq) == 2
        assert pq.extract_min()["item_id"] == "a"
        assert pq.extract_max()["item_id"] == "c"

    def test_delete_min_and_max(self, pq):
        pq.insert("min_elem", 1)
        pq.insert("mid_elem", 5)
        pq.insert("max_elem", 10)
        pq.delete("min_elem")
        assert pq.peek()["item_id"] == "mid_elem"
        pq.delete("max_elem")
        assert pq.peek()["item_id"] == "mid_elem"
        assert len(pq) == 1

    def test_delete_nonexistent_raises(self, pq):
        with pytest.raises(KeyError, match="not found"):
            pq.delete("missing")

    def test_delete_until_empty(self, pq):
        items = [f"item-{i}" for i in range(20)]
        for i, item_id in enumerate(items):
            pq.insert(item_id, i)
        for item_id in items:
            pq.delete(item_id)
        assert pq.is_empty()


# ======================================================================
# Persistence
# ======================================================================

class TestPersistence:

    def test_save_and_reload_roundtrip(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("task-1", 3, {"payload": 100})
        pq1.insert("task-2", 1, {"payload": 200})
        pq1.insert("task-3", 5, {"payload": 300})
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 3
        assert pq2.peek() == {"item_id": "task-2", "priority": 1, "value": {"payload": 200}}
        assert pq2.extract_max() == {"item_id": "task-3", "priority": 5, "value": {"payload": 300}}

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 10, 100, 1000])
    def test_reload_various_heap_sizes(self, tmp_path, size):
        storage = str(tmp_path / f"queue_{size}.json")
        pq1 = PersistentPriorityQueue(storage_path=storage)
        rng = random.Random(size)

        items = {f"i-{i}": rng.randint(-1000, 1000) for i in range(size)}
        for item_id, priority in items.items():
            pq1.insert(item_id, priority)
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        pq2._validate_invariants()
        assert len(pq2) == size
        assert pq2.peek()["priority"] == min(items.values())

    def test_persist_after_update_and_delete(self, storage):
        pq1 = PersistentPriorityQueue(storage_path=storage)
        pq1.insert("a", 10)
        pq1.insert("b", 20)
        pq1.insert("c", 30)
        pq1.update("c", 1)
        pq1.delete("a")
        del pq1

        pq2 = PersistentPriorityQueue(storage_path=storage)
        assert len(pq2) == 2
        assert pq2.peek()["item_id"] == "c"
        assert pq2.extract_max()["item_id"] == "b"

    def test_missing_file_starts_empty(self, tmp_path):
        missing_path = str(tmp_path / "does_not_exist" / "queue.json")
        pq = PersistentPriorityQueue(storage_path=missing_path)
        assert pq.is_empty()

    def test_stale_tmp_file_ignored(self, tmp_path):
        real_file = tmp_path / "queue.json"
        tmp_file = tmp_path / "queue.json.tmp"

        pq1 = PersistentPriorityQueue(storage_path=str(real_file))
        pq1.insert("real", 10)
        del pq1

        tmp_file.write_text(
            json.dumps({"version": 1, "next_seq": 5, "items": [{"item_id": "fake", "priority": 1, "seq": 0, "value": None}]}),
            encoding="utf-8",
        )

        pq2 = PersistentPriorityQueue(storage_path=str(real_file))
        assert pq2.peek()["item_id"] == "real"


# ======================================================================
# Validation & Mutation Safety
# ======================================================================

class TestValidationAndSafety:

    def test_duplicate_id_raises(self, pq):
        pq.insert("dup", 1)
        with pytest.raises(ValueError, match="already exists"):
            pq.insert("dup", 2)

    def test_invalid_priorities_raise(self, pq):
        for invalid_pri in [True, False, "1", None, float("nan"), float("inf"), float("-inf")]:
            with pytest.raises((TypeError, ValueError)):
                pq.insert("item", invalid_pri)

    def test_invalid_item_id_raises(self, pq):
        for invalid_id in [123, None, ("id",), ["id"]]:
            with pytest.raises(TypeError, match="string"):
                pq.insert(invalid_id, 1)

    def test_unserializable_value_does_not_mutate_state(self, pq):
        class Unserializable:
            pass

        assert len(pq) == 0
        with pytest.raises(TypeError, match="JSON-serializable"):
            pq.insert("bad_item", 1, value=Unserializable())
        assert len(pq) == 0
        assert "bad_item" not in pq._index

    def test_malformed_json_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not: valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted persistence file"):
            PersistentPriorityQueue(storage_path=str(bad_json))

    def test_unsupported_version_raises(self, tmp_path):
        bad_ver = tmp_path / "version.json"
        bad_ver.write_text(json.dumps({"version": 2, "next_seq": 0, "items": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported schema version"):
            PersistentPriorityQueue(storage_path=str(bad_ver))

    def test_inconsistent_next_seq_raises(self, tmp_path):
        bad_seq = tmp_path / "seq.json"
        bad_seq.write_text(
            json.dumps({
                "version": 1,
                "next_seq": 1,
                "items": [{"item_id": "a", "priority": 1, "seq": 5, "value": None}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="seq for 'a' must be an integer"):
            PersistentPriorityQueue(storage_path=str(bad_seq))


# ======================================================================
# Invariant Validation
# ======================================================================

class TestInvariants:

    def test_heap_and_index_invariants_under_load(self, pq):
        rng = random.Random(42)
        for i in range(100):
            pq.insert(f"item-{i}", rng.randint(-500, 500))
        pq._validate_invariants()

        for _ in range(25):
            pq.extract_min()
            pq._validate_invariants()

        for _ in range(25):
            pq.extract_max()
            pq._validate_invariants()

        for item_id in list(pq._index.keys())[:20]:
            pq.update(item_id, rng.randint(-500, 500))
            pq._validate_invariants()


# ======================================================================
# Randomized Differential Testing
# ======================================================================

class TestRandomizedDifferential:
    """Compare implementation against an independent reference model across randomized operation sequences."""

    def test_randomized_differential_against_oracle(self, tmp_path):
        storage = str(tmp_path / "diff_queue.json")
        pq = PersistentPriorityQueue(storage_path=storage)

        ref: dict[str, tuple] = {}
        next_seq = 0
        id_counter = 0
        rng = random.Random(1337)

        ops = ["insert", "extract_min", "extract_max", "peek", "is_empty", "update", "delete"]
        weights = [35, 15, 15, 10, 5, 10, 10]

        for step in range(25_000):
            op = rng.choices(ops, weights=weights, k=1)[0]

            if op == "insert":
                item_id = f"id-{id_counter}"
                id_counter += 1
                priority = rng.randint(-1000, 1000)
                value = f"v-{item_id}"
                pq.insert(item_id, priority, value)
                ref[item_id] = (priority, next_seq, value)
                next_seq += 1

            elif op == "extract_min":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.extract_min()
                else:
                    res = pq.extract_min()
                    expected_id = min(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert res["item_id"] == expected_id
                    assert res["priority"] == ref[expected_id][0]
                    assert res["value"] == ref[expected_id][2]
                    del ref[expected_id]

            elif op == "extract_max":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.extract_max()
                else:
                    res = pq.extract_max()
                    expected_id = max(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert res["item_id"] == expected_id
                    assert res["priority"] == ref[expected_id][0]
                    assert res["value"] == ref[expected_id][2]
                    del ref[expected_id]

            elif op == "peek":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.peek()
                else:
                    res = pq.peek()
                    expected_id = min(ref, key=lambda k: (ref[k][0], ref[k][1]))
                    assert res["item_id"] == expected_id
                    assert res["priority"] == ref[expected_id][0]

            elif op == "is_empty":
                assert pq.is_empty() == (len(ref) == 0)

            elif op == "update":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.update("missing", 0)
                else:
                    target = rng.choice(list(ref.keys()))
                    new_pri = rng.randint(-1000, 1000)
                    old_seq, old_val = ref[target][1], ref[target][2]
                    pq.update(target, new_pri)
                    ref[target] = (new_pri, old_seq, old_val)

            elif op == "delete":
                if not ref:
                    with pytest.raises(KeyError):
                        pq.delete("missing")
                else:
                    target = rng.choice(list(ref.keys()))
                    res = pq.delete(target)
                    assert res["item_id"] == target
                    assert res["priority"] == ref[target][0]
                    del ref[target]

            if step % 1000 == 0:
                pq._validate_invariants()
                assert len(pq) == len(ref)

        pq._validate_invariants()
        assert len(pq) == len(ref)

        # Drain remaining items in min order and verify match
        ref_sorted = sorted(ref.values(), key=lambda x: (x[0], x[1]))
        extracted = [pq.extract_min() for _ in range(len(pq))]
        for actual, (pri, seq, val) in zip(extracted, ref_sorted):
            assert actual["priority"] == pri
            assert actual["value"] == val
