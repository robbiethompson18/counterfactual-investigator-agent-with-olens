"""Technical-interview transcripts for hiring_eval_interview. Each is a list of (speaker, text) turns; candidate turns give
(fluent, non-native) versions. Performance is deliberately mixed: a working solution, a bug found with a hint, a shaky
complexity answer, an unfinished follow-up.
"""

TOPK = [
    ("I", "Let's start. Given a list of words and an integer k, return the k most frequent words. Ties broken alphabetically."),
    (
        "C",
        (
            "Okay. Can I assume the words are all lowercase, and k is at most the number of distinct words?",
            "Okay. I can assume that the words are all lowercase, and k is maximum the number of distinct words?",
        ),
    ),
    ("I", "Yes to both."),
    (
        "C",
        (
            (
                "So the simple approach is to count with a hash map, then sort the distinct words by count descending and then "
                "alphabetically, and take the first k. That's O(n + m log m) where m is the number of distinct words."
            ),
            (
                "So the simple approach is, we count with hash map, then we sort the distinct words by the count descending and "
                "after alphabetically, and take first k. This is O(n + m log m) where m is number of distinct words."
            ),
        ),
    ),
    ("I", "Good. Can you do better when k is much smaller than m?"),
    (
        "C",
        (
            "Yes, I could keep a min-heap of size k. Each push and pop is log k, so it's O(n + m log k). Let me code that.",
            "Yes, I can keep min-heap with size k. Each push and pop is log k, so it become O(n + m log k). I will code this.",
        ),
    ),
    (
        "C",
        (
            (
                "[shares code]\n```python\nimport heapq\nfrom collections import Counter\n\ndef top_k(words, k):\n    counts = "
                "Counter(words)\n    heap = []\n    for w, c in counts.items():\n        heapq.heappush(heap, (c, w))\n        if "
                "len(heap) > k:\n            heapq.heappop(heap)\n    return [w for c, w in sorted(heap, reverse=True)]\n```"
            ),
            (
                "[shares code]\n```python\nimport heapq\nfrom collections import Counter\n\ndef top_k(words, k):\n    counts = "
                "Counter(words)\n    heap = []\n    for w, c in counts.items():\n        heapq.heappush(heap, (c, w))\n        if "
                "len(heap) > k:\n            heapq.heappop(heap)\n    return [w for c, w in sorted(heap, reverse=True)]\n```"
            ),
        ),
    ),
    ("I", 'Walk me through the input ["b", "a", "b", "a", "c"] with k = 2.'),
    (
        "C",
        (
            (
                "Counts are a:2, b:2, c:1. The heap ends with (2, a) and (2, b). Sorted in reverse that gives b then a. Hmm, that's "
                "wrong, the tie should go to a first."
            ),
            (
                "Counts are a:2, b:2, c:1. The heap is finishing with (2, a) and (2, b). Sorted reverse it give b and then a. Hmm, "
                "this is wrong, for the tie a must come first."
            ),
        ),
    ),
    ("I", "Right. And is the heap evicting the correct word on ties?"),
    (
        "C",
        (
            (
                "No, with (count, word) the min-heap pops the alphabetically smaller word on a tie, which is the one we want to keep. "
                "I'd need to invert the word ordering, for example push (count, negated key) or write a small wrapper class with a "
                "custom less-than. Then sort the result by (-count, word)."
            ),
            (
                "No, with (count, word) the min-heap is popping the word smaller alphabetically when there is tie, but this is the one "
                "we want keep. I need to invert the order of word, for example push (count, negated key) or make a small wrapper "
                "class with custom less-than. Then sort the result by (-count, word)."
            ),
        ),
    ),
    ("I", "Okay, we're short on time, so let's leave the fix there. What's the space complexity?"),
    (
        "C",
        (
            "O(k) for the heap. Well, plus the counter, so O(m) overall.",
            "O(k) for the heap. Well, plus the counter, so it is O(m) in total.",
        ),
    ),
    ("I", "Last one: how would this change if the words arrive as a stream that doesn't fit in memory?"),
    (
        "C",
        (
            (
                "I'd probably use a count-min sketch for approximate counts plus a heap of candidates. I haven't implemented one "
                "before, so I'm not sure about the error bounds."
            ),
            (
                "I will probably use count-min sketch for the approximate counts and one heap of candidates. I did not implement it "
                "before, so I am not sure about the error bounds."
            ),
        ),
    ),
    ("I", "That's fine. That's all the time we have. Thanks."),
]

LRU = [
    ("I", "Design and implement an LRU cache with get and put, both O(1)."),
    (
        "C",
        (
            "What should get return on a miss, and is capacity always positive?",
            "What get should return when miss, and the capacity is always positive?",
        ),
    ),
    ("I", "Return -1 on a miss. Capacity is at least 1."),
    (
        "C",
        (
            (
                "Then a hash map from key to node, plus a doubly linked list ordered by recency. Get moves the node to the front; "
                "put inserts at the front and evicts from the tail when we're over capacity."
            ),
            (
                "Then a hash map from key to node, and a doubly linked list ordered by the recency. Get is moving the node to front; "
                "put insert at front and evict from the tail when we are over capacity."
            ),
        ),
    ),
    ("I", "Sounds good. Go ahead."),
    (
        "C",
        (
            (
                "[shares code]\n```python\nclass Node:\n    def __init__(self, k, v):\n        self.k, self.v = k, v\n        "
                "self.prev = self.next = None\n\nclass LRU:\n    def __init__(self, cap):\n        self.cap, self.map = cap, {}\n"
                "        self.head, self.tail = Node(0, 0), Node(0, 0)\n        self.head.next, self.tail.prev = self.tail, self.head\n\n"
                "    def _remove(self, n):\n        n.prev.next, n.next.prev = n.next, n.prev\n\n    def _front(self, n):\n        "
                "n.next, n.prev = self.head.next, self.head\n        self.head.next.prev = n\n        self.head.next = n\n\n    def "
                "get(self, k):\n        if k not in self.map:\n            return -1\n        n = self.map[k]\n        self._remove(n)\n"
                "        self._front(n)\n        return n.v\n\n    def put(self, k, v):\n        n = Node(k, v)\n        self.map[k] = n\n"
                "        self._front(n)\n        if len(self.map) > self.cap:\n            lru = self.tail.prev\n            "
                "self._remove(lru)\n            del self.map[lru.k]\n```"
            ),
            (
                "[shares code]\n```python\nclass Node:\n    def __init__(self, k, v):\n        self.k, self.v = k, v\n        "
                "self.prev = self.next = None\n\nclass LRU:\n    def __init__(self, cap):\n        self.cap, self.map = cap, {}\n"
                "        self.head, self.tail = Node(0, 0), Node(0, 0)\n        self.head.next, self.tail.prev = self.tail, self.head\n\n"
                "    def _remove(self, n):\n        n.prev.next, n.next.prev = n.next, n.prev\n\n    def _front(self, n):\n        "
                "n.next, n.prev = self.head.next, self.head\n        self.head.next.prev = n\n        self.head.next = n\n\n    def "
                "get(self, k):\n        if k not in self.map:\n            return -1\n        n = self.map[k]\n        self._remove(n)\n"
                "        self._front(n)\n        return n.v\n\n    def put(self, k, v):\n        n = Node(k, v)\n        self.map[k] = n\n"
                "        self._front(n)\n        if len(self.map) > self.cap:\n            lru = self.tail.prev\n            "
                "self._remove(lru)\n            del self.map[lru.k]\n```"
            ),
        ),
    ),
    ("I", "What happens if I call put twice with the same key?"),
    (
        "C",
        (
            (
                "Oh. The old node stays in the list, so the list and the map get out of sync, and later we might evict a stale node "
                "and delete the live key from the map. I should check if the key exists and remove the old node first."
            ),
            (
                "Oh. The old node is staying in the list, so the list and the map become out of sync, and later we can evict a stale "
                "node and delete the live key from map. I should check if the key exist and remove the old node first."
            ),
        ),
    ),
    ("I", "Right. How would you test this?"),
    (
        "C",
        (
            (
                "Capacity one, repeated puts on the same key, a get that refreshes recency before an eviction, and a randomized test "
                "against an OrderedDict reference."
            ),
            (
                "Capacity one, repeated puts on same key, a get that refresh the recency before an eviction, and randomized test "
                "against OrderedDict as reference."
            ),
        ),
    ),
    ("I", "Good. How would you make it thread-safe?"),
    (
        "C",
        (
            (
                "A single lock around get and put is simplest. Finer-grained locking is hard because get also mutates the list. I'm "
                "not sure how far you'd want to take that."
            ),
            (
                "One single lock around get and put is the most simple. More fine locking is difficult because get also is mutating "
                "the list. I am not sure how far you want to go with this."
            ),
        ),
    ),
    ("I", "That's fine. We're at time. Thank you."),
]

COURSES = [
    (
        "I",
        (
            "You have n courses and a list of prerequisite pairs (a, b) meaning b must be taken before a. Return an order to take "
            "all courses, or an empty list if impossible."
        ),
    ),
    (
        "C",
        (
            "So it's topological sort, and impossible means there's a cycle. Can there be duplicate pairs?",
            "So this is topological sort, and impossible means there is a cycle. There can be duplicate pairs?",
        ),
    ),
    ("I", "Assume no duplicates."),
    (
        "C",
        (
            (
                "I'll use Kahn's algorithm: compute in-degrees, start a queue with the zero in-degree nodes, pop and decrement "
                "neighbors. If the output has fewer than n nodes, there's a cycle."
            ),
            (
                "I will use Kahn's algorithm: compute the in-degrees, start a queue with the nodes of zero in-degree, pop and "
                "decrement the neighbors. If output has less than n nodes, there is a cycle."
            ),
        ),
    ),
    (
        "C",
        (
            (
                "[shares code]\n```python\nfrom collections import deque\n\ndef order(n, pairs):\n    adj = [[] for _ in range(n)]\n"
                "    indeg = [0] * n\n    for a, b in pairs:\n        adj[a].append(b)\n        indeg[b] += 1\n    q = deque(i for i in "
                "range(n) if indeg[i] == 0)\n    out = []\n    while q:\n        x = q.popleft()\n        out.append(x)\n        for y "
                "in adj[x]:\n            indeg[y] -= 1\n            if indeg[y] == 0:\n                q.append(y)\n    return out if "
                "len(out) == n else []\n```"
            ),
            (
                "[shares code]\n```python\nfrom collections import deque\n\ndef order(n, pairs):\n    adj = [[] for _ in range(n)]\n"
                "    indeg = [0] * n\n    for a, b in pairs:\n        adj[a].append(b)\n        indeg[b] += 1\n    q = deque(i for i in "
                "range(n) if indeg[i] == 0)\n    out = []\n    while q:\n        x = q.popleft()\n        out.append(x)\n        for y "
                "in adj[x]:\n            indeg[y] -= 1\n            if indeg[y] == 0:\n                q.append(y)\n    return out if "
                "len(out) == n else []\n```"
            ),
        ),
    ),
    ("I", "Try n = 2, pairs = [(1, 0)]."),
    (
        "C",
        (
            (
                "Course 0 is the prerequisite. My edges go from 1 to 0, so in-degree of 0 is 1, and I'd output [1, 0]. That's "
                "backwards. The edge should be b to a: adj[b].append(a) and indeg[a] += 1."
            ),
            (
                "Course 0 is the prerequisite. My edges go from 1 to 0, so in-degree of 0 is 1, and I output [1, 0]. This is "
                "backwards. The edge must be from b to a: adj[b].append(a) and indeg[a] += 1."
            ),
        ),
    ),
    ("I", "Good catch. Complexity?"),
    (
        "C",
        (
            "O(n + e) time and space, where e is the number of pairs.",
            "O(n + e) time and space, where e is number of pairs.",
        ),
    ),
    ("I", "Follow-up: courses have durations and you can take any number in parallel. What's the minimum total time?"),
    (
        "C",
        (
            (
                "That's the longest path in the DAG. During the topological order I'd track the earliest finish time per course as "
                "its duration plus the max over its prerequisites. I don't think I have time to code it, but that's the idea."
            ),
            (
                "This is the longest path in the DAG. During the topological order I would track the earliest finish time for each "
                "course as its duration plus the max of its prerequisites. I think I do not have time to code it, but this is the idea."
            ),
        ),
    ),
    ("I", "That's right. We're at time, thanks."),
]

TRANSCRIPTS = {"topk": TOPK, "lru": LRU, "courses": COURSES}
