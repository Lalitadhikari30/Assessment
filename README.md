# Persistent Priority Queue

A robust, production-grade **Double-Ended Persistent Priority Queue** implemented in pure Python (zero external dependencies). Backed by an **Indexed Min-Max Heap** (Atkinson et al., 1986), this data structure supports $O(\log n)$ dual min/max extractions, arbitrary item updates, and deletions, while persisting state atomically to disk.

---

## 🚀 Key Features

- **Double-Ended Priority Queue**: Efficiently access and extract both highest-priority (minimum value) and lowest-priority (maximum value) elements in $O(\log n)$ time.
- **$O(1)$ Positional Lookups**: Internal dictionary index mapping `item_id → heap_index` enables $O(\log n)$ updates and deletions of arbitrary elements without linear scans.
- **Strict FIFO Tie-Breaking**: Items with identical priority values retain strict FIFO (First-In, First-Out) ordering based on their initial insertion order. Priority updates preserve the original insertion order timestamp.
- **Crash-Resilient Atomic Persistence**: Uses an atomic write strategy (`write-to-temp` $\to$ `fsync` $\to$ `os.replace`) to guarantee data integrity even during sudden crashes or power interruptions.
- **Type-Safe Validation**: Strict runtime validation of input types and values (rejects `bool`, `NaN`, $\pm\infty$, duplicate IDs).

---

## ⏱️ Complexity & Operations

| Operation | Description | Time Complexity | Space Complexity |
| :--- | :--- | :---: | :---: |
| `insert(item_id, priority, value=None)` | Inserts a new item with unique ID and priority | $O(\log n)$ | $O(1)$ |
| `extract_min()` | Removes and returns item with lowest priority value | $O(\log n)$ | $O(1)$ |
| `extract_max()` | Removes and returns item with highest priority value | $O(\log n)$ | $O(1)$ |
| `peek()` | Inspects lowest priority item without removing it | $O(1)$ | $O(1)$ |
| `update(item_id, new_priority)` | Modifies priority of an existing item | $O(\log n)$ | $O(1)$ |
| `delete(item_id)` | Removes an arbitrary item by its ID | $O(\log n)$ | $O(1)$ |
| `is_empty()` | Checks if the priority queue contains 0 items | $O(1)$ | $O(1)$ |

---

## 🛠️ Architecture & Implementation Details

1. **Min-Max Heap Structure**:
   - Alternating levels in the binary tree represent *min-levels* (even depth $0, 2, \dots$) and *max-levels* (odd depth $1, 3, \dots$).
   - A node on a min-level is smaller than or equal to all descendants.
   - A node on a max-level is greater than or equal to all descendants.
   - The minimum element is always at root index `0`. The maximum element is always at index `1` or `2` (the children of the root).

2. **Positional Indexing (`_index`)**:
   - Maintains a bidirectional mapping `_index: dict[str, int]` kept in sync on every heap swap.
   - Allows arbitrary element deletion and priority updates to jump directly to the heap node in $O(1)$ time, followed by $O(\log n)$ sift operations (`_push_up` / `_push_down`).

3. **Atomic Persistence**:
   - State is stored in standard JSON schema with schema versioning and sequential metadata.
   - Prior to modifying the target file, data is written to a `.tmp` file, flushed to disk via `os.fsync()`, and replaced atomically using filesystem-level primitives (`os.replace`).

---

## 🌍 Real-World Use Cases

Priority queues are fundamental building blocks in high-scale systems:

1. **Workforce Management & Job Scheduling (e.g., SARALWEB)**:
   - Dynamic dispatching of field technicians or shift workers where urgent attendance, compliance alerts, and emergency service requests must preempt standard background tasks.
2. **Dijkstra’s Algorithm & A\* Pathfinding**:
   - Route planning, logistics optimization, and network packet routing where vertices with minimal tentative distance are expanded first.
3. **Operating System Process & Thread Schedulers**:
   - Multilevel feedback queues where CPU burst priorities determine process execution order, with real-time tasks taking precedence over batch jobs.
4. **Event-Driven Simulation & Timers**:
   - Simulators (e.g., network packet simulators, financial order-book matching engines) scheduling future events ordered by timestamp.
5. **Bandwidth Management & Quality of Service (QoS)**:
   - Routers prioritizing latency-sensitive voice/video traffic (VoIP) over bulk file transfer packets.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+ (Zero third-party runtime dependencies).
- `pytest` (for running the test suite).

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Example

```python
from module import PersistentPriorityQueue

# Initialize queue with automatic persistence to 'data/queue.json'
pq = PersistentPriorityQueue(storage_path="data/queue.json")

# 1. Insert tasks (lower number = higher urgency)
pq.insert("task-101", priority=5, value={"title": "Generate Monthly Report"})
pq.insert("task-102", priority=1, value={"title": "Emergency Server Restart"})
pq.insert("task-103", priority=8, value={"title": "Routine DB Backup"})

# 2. Peek at the most urgent task
print(pq.peek())
# Output: {'item_id': 'task-102', 'priority': 1, 'value': {'title': 'Emergency Server Restart'}}

# 3. Extract highest priority (minimum value)
urgent = pq.extract_min()
print("Processed:", urgent["item_id"])  # task-102

# 4. Extract lowest priority (maximum value)
batch_job = pq.extract_max()
print("Deferred:", batch_job["item_id"])  # task-103

# 5. Update priority of an existing task
pq.update("task-101", new_priority=2)

# 6. Delete a specific task
deleted = pq.delete("task-101")
print("Deleted:", deleted["item_id"])

# 7. Check if empty
print("Is empty?", pq.is_empty())  # True
```

---

## 🧪 Running the Test Suite

The test suite includes 72 thorough unit tests, randomized differential oracle tests (30,000 operations), crash/restart stress tests, and invariant checks:

```bash
python -m pytest -v
```
