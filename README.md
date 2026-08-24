# Persistent Priority Queue

A persistent double-ended priority queue implementation in Python (Standard Library only), backed by an **Indexed Min-Max Heap** (Atkinson et al., 1986) and file-based JSON persistence.

---

## 1. Problem Overview

Standard priority queues typically support retrieving only the minimum (or maximum) element in $O(\log n)$ time. However, many real-world workloads require:
1. Accessing both high-priority and low-priority items without rebuilding the heap.
2. Updating or deleting arbitrary items in $O(\log n)$ time without linear $O(n)$ search overhead.
3. Durable persistence to disk so that state survives application restarts and crashes.
4. Deterministic FIFO ordering when priorities are equal.

---

## 2. Design Choices & Architecture

### Why Indexed Min-Max Heap?
A **Min-Max Heap** is a complete binary tree that alternates between *min-levels* (even depths: 0, 2, 4...) and *max-levels* (odd depths: 1, 3, 5...):
- Every node on a min-level is smaller than or equal to all its descendants.
- Every node on a max-level is larger than or equal to all its descendants.
- **Root (index 0)** holds the global minimum ($O(1)$ peek, $O(\log n)$ extraction).
- **Root's children (indices 1 & 2)** hold the global maximum ($O(\log n)$ extraction).

This provides dual-ended capabilities in a single contiguous array without maintaining two separate heaps or complex pointer structures.

### Role of the Index Map
Standard heaps require $O(n)$ linear scans to find an arbitrary element before updating or deleting it. This implementation maintains an internal bidirectional hash map:
$$\text{\_index}: \text{item\_id} \to \text{heap\_index}$$
The index map is updated during every heap swap ($O(1)$). This enables:
- Arbitrary priority updates in $O(\log n)$ time.
- Arbitrary item deletions in $O(\log n)$ time.
- Immediate duplicate ID rejection in $O(1)$ time.

### Strict FIFO Tie-Breaking
Each item is assigned a monotonic sequence number `seq` upon insertion. Heap comparisons use the key `(priority, seq)`.
- When priorities are equal, items with smaller `seq` (inserted earlier) are dequeued first by `extract_min`.
- When updating an item's priority via `update()`, its original `seq` is preserved.

---

## 3. Persistence Strategy

### Atomic Writes
State is persisted to a single JSON file on each mutating operation (`insert`, `update`, `delete`, `extract_min`, `extract_max`).
To prevent file corruption during crashes or mid-write interruptions:
1. The serialized payload is written to a temporary file (`<filename>.tmp`).
2. `os.fsync()` flushes file buffers to physical storage.
3. `os.replace()` atomically swaps the temporary file into the destination path.

### Logical Representation & Reconstruction
The persisted JSON stores the queue's logical elements (`item_id`, `priority`, `seq`, `value`) and `next_seq`. The in-memory heap order and hash index are derived structures rebuilt in $O(n)$ time on load using Floyd's bottom-up heap construction algorithm.

---

## 4. Complexity Analysis

It is important to distinguish the in-memory data structure complexity from the file serialization overhead.

### In-Memory Complexity
| Operation | Time Complexity | Space Complexity |
| :--- | :---: | :---: |
| `insert(item_id, priority, value=None)` | $O(\log n)$ | $O(1)$ |
| `extract_min()` | $O(\log n)$ | $O(1)$ |
| `extract_max()` | $O(\log n)$ | $O(1)$ |
| `peek()` | $O(1)$ | $O(1)$ |
| `update(item_id, new_priority)` | $O(\log n)$ | $O(1)$ |
| `delete(item_id)` | $O(\log n)$ | $O(1)$ |
| `is_empty()` | $O(1)$ | $O(1)$ |

### Persistence Overhead
Because the entire queue is serialized to JSON on each mutation to guarantee crash consistency, disk persistence adds an **$O(n)$** serialization and I/O cost per mutation.

---

## 5. API & Usage Examples

```python
from module import PersistentPriorityQueue

# Initialize or load existing queue from disk
pq = PersistentPriorityQueue(storage_path="data/queue.json")

# 1. Insert items (numeric priority, optional JSON-serializable value)
pq.insert("job-101", priority=10, value={"task": "Generate report"})
pq.insert("job-102", priority=1, value={"task": "Critical server patch"})
pq.insert("job-103", priority=25, value={"task": "Nightly backup"})

# 2. Inspect the highest urgency item without removing it
print(pq.peek())
# Output: {'item_id': 'job-102', 'priority': 1, 'value': {'task': 'Critical server patch'}}

# 3. Extract minimum and maximum elements
urgent_job = pq.extract_min()  # Returns job-102 (priority 1)
batch_job = pq.extract_max()   # Returns job-103 (priority 25)

# 4. Update priority of an existing item
pq.update("job-101", new_priority=2)

# 5. Delete an arbitrary item
deleted = pq.delete("job-101")

# 6. Check status
print("Is empty:", pq.is_empty())
```

---

## 6. Real-World Use Cases

1. **Workforce Management & Job Dispatch (e.g., SARALWEB)**:
   - Scheduling field staff and worker assignments where critical emergency calls preempt scheduled shifts, while low-priority routine tasks are deferred when load is high.
2. **Graph Routing & Pathfinding (Dijkstra / A\*)**:
   - Expanding graph nodes with lowest estimated cost while updating path distances (`update()`) when shorter paths are discovered.
3. **Operating System Scheduling**:
   - Prioritizing real-time interactive tasks over background batch computation with deterministic tie-breaking.
4. **Network Bandwidth & Quality of Service (QoS)**:
   - Prioritizing latency-sensitive VoIP/video packets over bulk data downloads.

---

## 7. Assumptions & Input Constraints

- **IDs**: Must be non-empty strings and unique within the queue.
- **Priorities**: Must be finite `int` or `float` (`bool`, `NaN`, and $\pm\infty$ are rejected).
- **Values**: Optional payload must be JSON-serializable. Value serializability is validated before modifying in-memory state.
- **Dependencies**: Python standard library only for runtime. `pytest` is used for testing.

---

## 8. Setup & Testing

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
Run the test suite including unit tests, invariant checks, persistence reload across varying queue sizes, and randomized differential testing:

```bash
python -m pytest -v
```
