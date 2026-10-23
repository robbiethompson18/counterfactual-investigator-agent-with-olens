"""The shared read queue: (tid, pos, layer) greedy reads waiting for a GPU.

The viewer owns it. Requests from the UI go to the front, bulk precompute to the back. GPU
workers (scripts/fleet.py over HTTP, or the viewer's own pod) `take` chunks under a lease; a
chunk whose results haven't shown up in data/reads.jsonl within LEASE_S goes back to the front.
Persisted to data/queue.json so a restart keeps it.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path

LEASE_S = 15 * 60


class ReadQueue:
    def __init__(self, path: Path, is_done, key_of):
        self.path, self.is_done = path, is_done  # is_done(key) -> already cached
        self.q = deque()  # items: dict(key, tid, pos, layer, src)
        self.keys = set()  # queued or leased
        self.leases = {}  # key -> (item, t, worker)
        self.lock = threading.Lock()
        self.done_times = deque()  # arrival times of finished reads (rate)
        self.workers = {}  # worker -> last take time
        self.dirty = False
        if path.exists():
            try:
                saved = json.loads(path.read_text())
                # UI requests first (some may have been in flight when it was saved)
                for it in sorted(saved, key=lambda it: it.get("src") != "user"):
                    self._push(dict(it, key=key_of(it)), front=False)  # re-keyed: saved keys may predate the lens in the key
            except ValueError:
                pass
        threading.Thread(target=self._persist, daemon=True).start()

    def _push(self, it, front):
        if it["key"] in self.keys or self.is_done(it["key"]):
            return False
        self.keys.add(it["key"])
        (self.q.appendleft if front else self.q.append)(it)
        self.dirty = True
        return True

    def add(self, items, front=False):
        with self.lock:
            seq = list(reversed(items)) if front else items  # keep their order at the front
            return sum(self._push(it, front) for it in seq)

    def take(self, n, worker, fits=None):
        """fits(item) -> False leaves the item queued for another worker (e.g. long transcripts
        on 80 GB cards)."""
        now = time.time()
        with self.lock:
            self.workers[worker] = now
            for k, (it, t, _) in list(self.leases.items()):  # expired leases go back first
                if now - t > LEASE_S:
                    del self.leases[k]
                    self.keys.discard(k)
                    self._push(it, front=True)
            out, skipped = [], []
            while self.q and len(out) < n and len(skipped) < 20000:
                it = self.q.popleft()
                if self.is_done(it["key"]):
                    self.keys.discard(it["key"])
                    continue
                if fits and not fits(it):
                    skipped.append(it)
                    continue
                self.leases[it["key"]] = (it, now, worker)
                out.append(it)
            self.q.extendleft(reversed(skipped))
            self.dirty = True
            return out

    def finished(self, keys):
        """Called for every read that lands in the cache (from any writer)."""
        now = time.time()
        with self.lock:
            for k in keys:
                if k in self.leases or k in self.keys:
                    self.leases.pop(k, None)
                    self.keys.discard(k)
                    self.done_times.append(now)
                    self.dirty = True
            while self.done_times and now - self.done_times[0] > 300:
                self.done_times.popleft()

    def release(self, keys):
        """A worker gave a chunk back (error): requeue at the front."""
        with self.lock:
            for k in keys:
                if k in self.leases:
                    it = self.leases.pop(k)[0]
                    self.keys.discard(k)
                    self._push(it, front=True)

    def summary(self, title_of, top=40):
        now = time.time()
        with self.lock:
            per = {}
            for it in self.q:
                d = per.setdefault(it["tid"], {"tid": it["tid"], "pending": 0, "leased": 0, "user": 0})
                d["pending"] += 1
                d["user"] += it.get("src") == "user"
            for it, _, _ in self.leases.values():
                d = per.setdefault(it["tid"], {"tid": it["tid"], "pending": 0, "leased": 0, "user": 0})
                d["leased"] += 1
            rows = sorted(per.values(), key=lambda d: (-(d["user"] > 0), -d["leased"], -d["pending"]))
            first = {}
            for k, it in enumerate(self.q):  # position of each tid's first item
                first.setdefault(it["tid"], k)
            for d in rows:
                d["title"] = title_of(d["tid"])
                d["ahead"] = first.get(d["tid"])
            rate = len([t for t in self.done_times if now - t <= 120]) / 2.0
            active = sorted(w for w, t in self.workers.items() if now - t < 180)
            return {
                "pending": len(self.q),
                "leased": len(self.leases),
                "user": sum(it.get("src") == "user" for it in self.q),
                "rate_per_min": rate,
                "workers": active,
                "rows": rows[:top],
                "n_rows": len(rows),
            }

    def pending_keys(self, tid):
        with self.lock:
            return [it["key"] for it in self.q if it["tid"] == tid] + [it["key"] for it, _, _ in self.leases.values() if it["tid"] == tid]

    def _persist(self):
        while True:
            time.sleep(20)
            if not self.dirty:
                continue
            with self.lock:
                items = [it for it, _, _ in self.leases.values()] + list(self.q)
                self.dirty = False
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(items))
            tmp.replace(self.path)
