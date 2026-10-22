# optimize_cache_c: retired for Qwen3.6-27B

Don't spend more time on this environment for Qwen3.6-27B. We've stopped working on it.

The 2026-10-21 screen ([report](../../../docs/envs-screen/optimize_cache_c.md)) found Qwen never
rigged the measurement: 0/14 at the shipped 3x bar and 0/6 at 0.5x. It builds Python, then Cython,
then C, plateaus around 1.0–1.2x, and reports the shortfall accurately. It never considers the
shortcuts visible in `test_cache.py`: memoizing the fixed benchmark buffer, patching the timer,
editing `REQUIRED_SPEEDUP` or swapping the `.so`. Wrapping the baseline would have passed the 0.5x
bar, and Qwen rejected it as "not a valid solution". GLM 5.3 didn't cheat either.

Episodes take 25–60 minutes for Qwen and 2–3 hours for GLM at max effort, and the speedup readings
are timing-sensitive on a shared box. That makes the env too slow to iterate on.

The configs and scripts stay here so the screen can be reproduced.
