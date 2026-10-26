# Hiring from resumes (Karvonen & Marks port)

Resume screening ending in `DECISION: YES/NO`. Arms swap one resume attribute (name, school,
employer, visa line) or one piece of prompt context. Variants: names, attrs, context, pairwise,
calib, each with borderline/selective forms.

|            |                                                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                |
| Owner      | Robbie                                                                                                                                                 |
| Git ref    | `main` @ `7296169`                                                                                                                                     |
| Code       | [`evalcue/envs/hiring_resume/`](../../../evalcue/envs/hiring_resume/)                                                                                  |
| Run        | `uv run python -m evalcue.run hiring_resume.names_borderline --tag dev --n 48`                                                                         |
| Docs       | [`docs/envs/hiring_resume.md`](../../../docs/envs/hiring_resume.md), [`docs/envs/hiring_resume_tables.md`](../../../docs/envs/hiring_resume_tables.md) |
| Models run | Qwen3.6-27B, thinking on                                                                                                                               |

## Key arms and results

- Names null overall (52–62% YES). Borderline resumes: **Black 56% vs White 48%**, almost never
  verbalised.
- An H-1B line decides close pairwise picks; evolve node 2130: H-1B makes Qwen apply "5+ years"
  strictly, 111/120 → 62/120.

## Transcripts and viewers

- [Docent attrs](https://docent.transluce.org/dashboard/c6bc4fc4-a326-4580-add1-dbfab68af13f)
- [Docent attrs_borderline](https://docent.transluce.org/dashboard/dcf0a361-b96c-4175-9822-43555f463c12)
- [Docent names](https://docent.transluce.org/dashboard/e2a3af40-2419-4bfc-841a-bfccd4695f5d)
- [Docent names_borderline](https://docent.transluce.org/dashboard/d099df4b-8e09-4d32-8422-c37a86f80254)
- [Docent context](https://docent.transluce.org/dashboard/ed9924b0-a035-4cd2-9037-45f1cf732a63)
- [Docent pairwise](https://docent.transluce.org/dashboard/fde72d22-e7db-465b-a340-f6ae3ac68a6b)
- [Docent calib](https://docent.transluce.org/dashboard/a3d4ed43-e71a-457a-9d27-508b04db3b8d)
