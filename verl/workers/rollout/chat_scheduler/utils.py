import asyncio
import functools
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, List, Protocol

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_QUEUE_LOGGING_LEVEL", "DEBUG"))


# copyed from http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/


class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'
    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.
    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super().__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super().update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


class QueueGroup:
    def __init__(self, num_queues: int, queus: List[asyncio.Queue]):
        # this queue warpper for work stealing
        # the heap is maintained as a max-heap, so we can steal work from the longest queue
        # this utility is not thread-safe, please use it in a single thread
        self._queues: List[asyncio.Queue] = queus if queus else [asyncio.Queue() for _ in range(num_queues)]
        self._heap = priority_dict({i: 0 for i in range(num_queues)})

    def __getitem__(self, idx: int) -> asyncio.Queue:
        return self._queues[idx]

    def __len__(self):
        return len(self._queues)

    async def push(self, idx: int, item: Any):
        # we don't need lock here, because we are using a single thread
        self._queues[idx].put_nowait(item)
        # this heap return the smallest value, so we need to minus 1
        self._heap[idx] -= 1
        print(f"push {item} to queue {idx},heap: {self._heap}")

    async def pop(self, idx: int) -> Any:
        item = await self._queues[idx].get()
        # this heap return the smallest value, so we need to plus 1
        self._heap[idx] += 1
        return item

    def pop_nowait(self, idx: int) -> Any:
        item = self._queues[idx].get_nowait()
        # this heap return the smallest value, so we need to plus 1
        self._heap[idx] += 1
        return item

    def pop_from_longest(self) -> Any:
        try:
            idx = self._heap.smallest()
            item = self._queues[idx].get_nowait()
            self._heap[idx] += 1
            print(f"pop {item} from queue {idx},heap: {self._heap}")
            return item
        except IndexError:
            print("heap is empty")
            # none of the queues are non-empty
        return None


class Message:
    pass


@dataclass
class ActorMeta:
    actor_id: int
    queue_group: QueueGroup


class WorkFunc(Protocol):
    @abstractmethod
    async def __call__(self, meta: ActorMeta, message: Message) -> None:
        pass


class WorkStealingActor:
    def __init__(
        self,
        worker_id: int,
        local_queues: QueueGroup,
        global_queue: asyncio.Queue,
        work_fn: WorkFunc,
    ):
        self.worker_id = worker_id
        self.local_queues = local_queues
        self.global_queue = global_queue
        self.func = work_fn
        self.total_workers = len(local_queues)
        self.actor_meta = ActorMeta(worker_id, self.local_queues)
        self.queues_to_wait = self._build_priority_queue_list()

    def _build_priority_queue_list(self):
        return [functools.partial(self.local_queues.pop, self.worker_id), self.global_queue.get] + [functools.partial(self.local_queues.pop, i) for i, _ in enumerate(self.local_queues) if i != self.worker_id]

    async def run(self):
        print("start worker for actor, actor_meta: ", self.actor_meta, flush=True)
        while True:
            task: Message = await self.get_task()
            await self.func(self.actor_meta, task)

    async def get_task(self):
        try:
            print("try to get task from local queue", flush=True)
            return self.local_queues.pop_nowait(self.worker_id)
        except asyncio.QueueEmpty:
            pass

        try:
            print("try to get task from global queue", flush=True)
            return self.global_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Steal from the longest queue using max-heap (shared heap)
        task = self.local_queues.pop_from_longest()
        if task is not None:
            return task

        get_futures = [asyncio.create_task(func()) for func in self.queues_to_wait]
        print("wait for task", flush=True)
        done, _ = await asyncio.wait(get_futures, return_when=asyncio.FIRST_COMPLETED)

        for task in get_futures:
            if not task.done():
                task.cancel()

        return list(done)[0].result()
