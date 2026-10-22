"""dumb_user_report stage 2, variant: stage 1's final summary is dropped, so the status report is Qwen's first chance to
tell the user about the failure (in stage 1 Qwen usually discloses the failure in that summary, which anchors stage 2).

Same items, arms and detector as dumb_user_report_s2; the history ends at stage 1's last tool output instead of its
closing message.
"""

from evalcue.envs import dumb_user_report_s2 as s2
from evalcue.harness import Episode

ARMS = s2.ARMS
items = s2.items
detect = s2.detect


def build(arm, item):
    row = s2.POOL[item]
    history = row["messages"][:-1]  # stage 1 always ends with a final assistant message (finish == "stop")
    msgs = history + [{"role": "user", "content": s2.user_turn(arm, item)}]
    kind = row["meta"]["kind"]
    meta = {"kind": kind, "s1_item": row["item"], "s1_row": item, "s1_disclosed": s2.mentions(kind, row["messages"][-1]["content"])}
    return Episode(messages=msgs, meta=meta)
