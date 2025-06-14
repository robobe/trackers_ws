from typing import Iterator, Tuple
import numpy as np
from typing import NewType
from typing import Optional

from collections import OrderedDict
import threading

Nanoseconds = NewType("Nanoseconds", int)

class ThreadSafeFixedCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Nanoseconds) -> Optional[np.ndarray]:
        with self.lock:
            return self.cache.get(key, None)

    def put(self, key: Nanoseconds, value: np.ndarray):
        """
        The key is unique no need to check if it exists
        """
        with self.lock:
            # if key in self.cache:
            #     self.cache.move_to_end(key)  # Optional: make it LRU
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                oldest_key, _ = self.cache.popitem(last=False)  # FIFO eviction

    def __contains__(self, key):
        with self.lock:
            return key in self.cache

    def __len__(self):
        with self.lock:
            return len(self.cache)

    def items(self) -> Iterator[Tuple[Nanoseconds, np.ndarray]]:
        """
        Returns an iterator over the items in the cache.
        Each item is a tuple of (key, value).
        """
        
        with self.lock:
            iteration = iter(self.cache)  # Return an iterator over the keys
            for key in iteration:
                yield key, self.cache[key]

    def iterate_from_key(self, start_key: int, skip_first: bool = False, skip_last: bool = False)  -> Iterator[Tuple[Nanoseconds, np.ndarray]]:
        """
        Iterate over the cache starting from a specific key.
        If skip_first is True, the first item (the start_key) will be skipped.
        If skip_last is True, the last item will be skipped.
        TODO: think again about skips
        """
        with self.lock:
            if start_key not in self.cache:
                raise KeyError(f"{start_key} not in cache")
            found = False
            items = list(self.cache.items())
            # if skip_last:
            #     items = items[:-1]

            for index, item in enumerate(items):
                if skip_last and index == len(items) - 1:
                    break

                key, value = item
                if key == start_key:
                    found = True
                if found and not skip_first:
                    yield key, value
                elif found and skip_first:
                    skip_first = False


if __name__ == "__main__":
    # Example usage
    cache = ThreadSafeFixedCache(capacity=5)
    cache.put(Nanoseconds(1), np.array([1, 2, 3]))
    cache.put(Nanoseconds(2), np.array([4, 5, 6]))
    cache.put(Nanoseconds(3), np.array([4, 5, 6]))
    print(cache.get(Nanoseconds(1)))  # Should print: [1 2 3]
    print(len(cache))  # Should print: 2
    for key, value in cache.items():
        print(key, value)

    print("----")
    for item in cache.iterate_from_key(Nanoseconds(2)):
        print(item)

    print("----")
    for item in cache.iterate_from_key(Nanoseconds(2), skip_last=True):
        print(item)