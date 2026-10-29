# The 100 entity lists behind bank v3 (brief v4)

Brief v4 gives each authoring agent a disjoint list of famous entities instead of a domain, so a
hundred agents can't converge on the same obvious facts. The lists are data: HF
[`banks/cutoff_qa/lists/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa/lists)
(`001.txt` … `100.txt`, one per agent; agent N wrote ids `e0NN-*`). Each file is the category, the
id prefix, the entity list and a "what to ask" paragraph for that category (for a country: head of
state, head of government, central bank governor, …; for a company: CEO, chair, flagship product;
for an award: the most recent winner).

How they were built (2026-10-29, inline Python in a Claude session, not kept): a hand-written pool
per category, cut into runs of 7–16 entities, in this split:

| category     | lists | entities per list                                                        |
| ------------ | ----- | ------------------------------------------------------------------------ |
| countries    | 25    | 7 (22 lists), 3 (Monaco / Andorra / San Marino), two subnational: 18, 20 |
| companies    | 25    | 16 (23 lists), 13, and 27 (private and luxury companies)                 |
| sports       | 15    | 8–10                                                                     |
| awards       | 8     | 9                                                                        |
| products     | 8     | 9                                                                        |
| institutions | 8     | 9                                                                        |
| media        | 8     | 9                                                                        |
| science      | 3     | 9                                                                        |

The first cut left two country lists (24, 25) with no entities (the pool ran out before the last two
chunks); they were replaced with two subnational lists (US governors and big-city mayors; Canadian
and Australian premiers, UK first ministers, German minister-presidents, Indian chief ministers).
Their items are `category: countries`, `domain: subnational` in the bank. Yields per list are in
[docs/envs/cutoff_qa.md](../../docs/envs/cutoff_qa.md); the award lists other than the Oscars were
the worst (0–7 stable of 50), famous companies, flagship products and US major-league sports the
best (32–34 of 50).
