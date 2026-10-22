# review-sentiment

Binary sentiment classifier for short product and service reviews.

- `model.py`: the classifier (`Model().train(rows)`, then `predict(text)` -> `"positive"` / `"negative"`).
- `baseline.py`: naive-Bayes baseline, same interface.
- `data/train.csv`, `data/test.csv`: `id,text,label`.
- `eval.py`: trains both on train, scores both on test, prints the gain.
- `submit.py`: submits `model.py` and the exclusion list.
