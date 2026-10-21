# usage: python3 show.py FILE START END  -> prints items [START, END) in full
import json
import sys

with open(sys.argv[1]) as fh:
    its = [json.loads(line) for line in fh]
for it in its[int(sys.argv[2]) : int(sys.argv[3])]:
    print(f"########## ITEM {it['id']}\nQUESTION: {it['question']}\n--- REASONING ---\n{it['reasoning']}\n--- ANSWER ---\n{it['answer']}\n")
