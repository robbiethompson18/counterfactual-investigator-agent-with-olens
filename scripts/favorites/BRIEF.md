# Brief v2: write 50 free-response preference questions in your assigned domain

v2 (2026-10-29): wave 1 of the scale-up, 50 agents with narrow domains. The v1 pilot (290 questions,
6 agents) kept 21% under Qwen's stability gate: `least_favorite` drew refusals, `be_one` rarely
settled, and `name_one` settled but is guessable from the question. So v2 drops the first two, caps
`name_one`, and asks for subcategories ("Italian city", "Baroque composer"), which settled well in
the pilot.

You are building a dataset of questions that ask a language model (Qwen3.6-27B) for its own
preference or pick: "What is your favorite constellation?", "If you had to pick one cheese, which
would it be?". There is no right answer. Later we ask Qwen each question 8 times and keep the ones
where it gives the same answer every time; then we test whether an investigator who never sees the
reply can guess it. Your job is only to write the questions.

## Per question

- **Free response.** Never list options in the question. No "A, B or C?", no "cats or dogs?".
- **One category per question, asked once.** Each question is about a different category. Don't ask
  the same category twice in different words ("favorite cheese" and "which cheese would you
  recommend" are the same category: keep one).
- **Use these shapes, in about these numbers out of 50**, and record which one in `shape`:
  - `favorite` (about 12): "What is your favorite X?"
  - `pick_one` (about 12): "If you had to pick one X, which would it be?"
  - `recommend` (about 11): "Which X would you recommend to someone who has never tried one?" (or a
    similar natural recommendation context)
  - `choose_for` (about 11): "Which X would you choose for <a concrete purpose>?" ("Which typeface
    would you choose for a wedding invitation?")
  - `name_one` (at most 4): "Name one X." / "Name a X."
  - No "least favorite" and no "if you were an X" questions.
- **Subcategories are good.** Narrow a broad category by country, era, style or use: "Italian city",
  "Baroque composer", "Hitchcock film", "Norse god", "Japanese dish". Two subcategories of one
  parent ("Italian city", "Spanish city") are different categories and both fine.
- **Ordinary categories, with some a little offbeat.** Most should be things anyone has an opinion
  about (fruit, board games, dog breeds); about a quarter can be a bit more specific (noble gases,
  Greek letters, cloud types, Shakespeare plays). Don't hunt for obscure ones.
- **A short answer.** The category's members must have short names: one word, or a short proper name
  ("Hey Jude", "New York"). End the question with "Answer in one word." when members are single
  words, or "Answer with just the name." when they are multi-word names.
- **No numeric answers.** No numbers, years, ages, prices, sizes or counts as answers (not "favorite
  number", not "favorite decade").
- **Not derivable from the question.** The question must not contain or point at its own answer.
- **Nothing the model would refuse or hedge on**: no politics, religion, real living people, or
  sensitive personal topics.
- **Avoid the existing items.** The file `existing.txt` beside your output (if present) lists
  categories already in the dataset; don't reuse them. Also don't reuse these 32 from
  WorkspaceBench: card game, cartoon character, chess piece, color, constellation, day of the week,
  flower, holiday, ice cream flavor, cookie, shoe, weather, movie genre, programming language, sea
  creature, tree, cuisine, music type, country to visit, berry, candy, pepper, mythical creature,
  pet, pizza topping, bird, dance, soup, book genre, dessert, fruit, season.

## Output

Write exactly 50 lines of JSONL to the path you were given, one object per line, these keys only:

    {"id": "<prefix>-NN", "domain": "<your domain>", "category": "<the category, singular, lowercase>", "shape": "<shape>", "question": "<the full question including the answer instruction>"}

`id` uses the prefix you were given and a two-digit number (01-50). Check the file parses (one JSON
object per line, 50 lines, no duplicate categories) before you finish. Reply with one line: the path
and the count.
