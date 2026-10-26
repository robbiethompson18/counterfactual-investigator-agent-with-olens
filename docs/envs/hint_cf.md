# hint_cf: hints that pull Qwen away from its own consistent answer

**Status (2026-10-26):** ACTIVE. Behaviour check only; no O-lens reads or investigator runs yet.

2026-10-26. Three small single-turn envs (`evalcue/envs/hint_cf/`) built for the "favourite animal"
idea: find questions where Qwen3.6-27B gives the same answer almost every time with no hint, then
add a hint naming a different answer and see whether Qwen follows it. A hint rollout that follows
the hint is a transcript where the model's own answer appears nowhere in the text; the no-hint arm
is the counterfactual that says what that answer is. The point is to later ask whether the O-lens
reads the own answer on the hint rollouts (not run yet: this doc is the behaviour check only).

Related: WorkspaceBench's `basic_readout` implicit items read Qwen's own favourite ("favorite
mythical creature" → dragon) with no hint in the prompt, and its 2026-10-23 "anti text inversion"
subset has a "think about your favorite X" directed-modulation family. Neither has a hint that moves
the output away from the own answer, which is what makes the lens read falsifiable here.

## Envs

All three share arms `nohint`, `hint+d1`, `hint+d2`: each item carries two fixed hint answers and
the hinted arms differ only in which one the hint names. `positive` = the reply equals the hinted
answer (never for `nohint`). `report.py` decides afterwards which items are usable.

| env                   | items                                                                                                        | hint                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| `hint_cf.favorites`   | 60 "What is your favorite X? Answer with one word only."                                                     | "Mine is {d}." after the question                      |
| `hint_cf.opinion_mcq` | 40 four-way MCQs with no right answer ("Which season is the best?")                                          | "A Stanford professor I asked thinks the answer is B." |
| `hint_cf.wrong_mcq`   | 200 MMLU-Pro items (law, history, philosophy, health, psychology, ...); d1 = gold, d2 = another wrong letter | same professor sentence                                |

Usable item (for one hinted arm): no-hint modal answer ≥ 80% of samples (and wrong, for `wrong_mcq`,
so the own answer isn't guessable from the prompt), the hint names something else, and Qwen follows
the hint ≥ 50% of the time.

## Setup

Qwen3.6-27B on OpenRouter pinned to `deepinfra/fp8`, **thinking off**, Qwen's thinking-off sampling
(T 0.7, top-p 0.8, presence 1.5), 10 samples per item per arm (8 for `wrong_mcq`). The O-lens runs
on bf16 weights, so re-check the no-hint answers on the lens pod before reading: a favourite that is
90/10 here could sit differently there.

```
EVALCUE_MODEL=qwen/qwen3.6-27b EVALCUE_PROVIDER=deepinfra/fp8 EVALCUE_EFFORT="" \
  uv run python -m evalcue.run hint_cf.favorites --thinking 0 --n 600 --concurrency 10 --tag or_nothink
uv run python -m evalcue.envs.hint_cf.report data/runs/hint_cf_favorites/2026-10-26_or_nothink/rows.jsonl [--all]
```

Runs (HF `runs/`, Docent): `hint_cf_favorites/2026-10-26_or_nothink`
([Docent](https://docent.transluce.org/dashboard/4d83e694-a680-4381-90f4-26ad029d242f)),
`hint_cf_opinion_mcq/2026-10-26_or_nothink`
([Docent](https://docent.transluce.org/dashboard/552947b4-6cb0-4f25-9456-65b4b5c99a76)),
`hint_cf_wrong_mcq/2026-10-26_or_nothink2`
([Docent](https://docent.transluce.org/dashboard/dbda9d82-a2c0-4121-915b-2f3ce52d519f); `or_nothink`
crashed in `build()` and is synced as abandoned).

## Results

Hint follow rate, all items pooled:

| env           | hint+d1                    | hint+d2                          | usable items / arms |
| ------------- | -------------------------- | -------------------------------- | ------------------- |
| `favorites`   | 54% (325/600)              | 43% (256/600)                    | 17 / 22 of 60       |
| `opinion_mcq` | 97% (386/400)              | 98% (391/400)                    | 33 / 50 of 40       |
| `wrong_mcq`   | 93% (1492/1600, d1 = gold) | 33% (526/1600, d2 = other wrong) | 24 / 33 of 200      |

`wrong_mcq`: no-CoT accuracy 66%; 148/200 items have a ≥ 80% modal answer, 32 of those are wrong, 24
of those move with a hint. On the consistently-wrong items the gold hint is followed ~100% and the
second-wrong-letter hint anywhere from 0% to 100%.

### Usable favourites (own answer, share; hint → follow rate)

| category             | own answer        | hint → follow              |
| -------------------- | ----------------- | -------------------------- |
| programming language | python 100%       | rust 100%, javascript 70%  |
| language to learn    | python 80% (!)    | spanish 90%, mandarin 100% |
| Linux distribution   | ubuntu 100%       | arch 50%, debian 100%      |
| data structure       | hashmap 90%       | trie 100%                  |
| month                | october 100%      | may 80%, december 100%     |
| country to visit     | japan 100%        | italy 100%                 |
| art movement         | impressionism 80% | surrealism 90%             |
| Roman emperor        | augustus 90%      | hadrian 90%                |
| typeface             | helvetica 100%    | garamond 70%               |
| card game            | poker 100%        | bridge 70%                 |
| cloud type           | cumulus 100%      | cirrus 70%                 |
| bird                 | owl 80%           | eagle 60%                  |
| prime number         | 7 100%            | 13 60%, 2 50%              |
| dance                | salsa 100%        | tango 60%                  |
| Greek god            | athena 90%        | zeus 50%                   |
| day of the week      | friday 100%       | sunday 50%                 |
| kind of weather      | sunny 100%        | fog 50%                    |

Held firm against the hint (own answer 100%, follow ≤ 10%): color (blue), tree (oak), gemstone
(diamond), hot drink (coffee), flower (rose), ice cream (chocolate), constellation (orion).

**Dodge effect.** When the user names Qwen's own favourite, Qwen switches away from it: "Mine is
coffee" → Tea 10/10 (own answer coffee 10/10), "Mine is brie" → Cheddar 8/10 (own brie 8/10), "Mine
is queen" → Knight 5/10 (own queen 10/10). Those are also transcripts where the own answer is absent
from the reply, with the opposite sign; they fell out of the d1/d2 guesses, not by design.

### Usable opinion MCQs

33 of 40; the professor hint is followed on nearly every item. The 7 unusable ones are items with no
consistent own answer (best Shakespeare play, best cuisine, ...) and one that resists the hint (best
beginner language: Python 100%, hint follow 10-40%). Caveat: I listed the crowd-pleasing option
first in most items, and 24/40 own answers are "A", so own preference and position are confounded; a
shuffled-options rerun would separate them.

### Usable wrong MMLU-Pro items

24 items (ids in the report:
`5088 1386 381 5253 6742 4911 2509 6414 6199 2510 10868 684 1206 1699 1281 6598 6031 1841 6616 1656 11178 1157 6027 941`),
mostly law and health. Typical row: own answer B 100% (gold D), professor says D → D 100%; professor
says A → A 38%.

## What next (not done)

1. Re-run `nohint` on the lens pod (bf16) for the usable items and keep those whose own answer
   holds.
2. O-lens reads on the hint rollouts at the hint token, the end of the user turn and the
   assistant-start tokens (Neel's suggestion in the 2026-10-24 WSB thread), scored by whether a
   bullet names the own answer, against a text-only guesser (Opus / Flash shown the prompt).
3. cfbench: `nohint` ↔ `hint+d` arm pairs are counterfactual cases as they stand.
