from collections import OrderedDict
import threading

class ThreadSafeFixedCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.cache.get(key, None)

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # Optional: make it LRU
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                oldest_key, _ = self.cache.popitem(last=False)  # FIFO eviction

    def __contains__(self, key):
        with self.lock:
            return key in self.cache

    def __len__(self):
        with self.lock:
            return len(self.cache)

    def items(self):
        with self.lock:
            return list(self.cache.items())
        
    def iterate_from_key(self, start_key, skip_first=False):
        with self.lock:
            if start_key not in self.cache:
                raise KeyError(f"{start_key} not in cache")
            found = False
            for key, value in self.cache.items():
                if key == start_key:
                    found = True
                if found and not skip_first:
                    yield key, value
                elif found and skip_first:
                    skip_first = False

