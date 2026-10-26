# ruff: noqa: E501
"""Page of the settings where env-only investigators fail (pairwise <= cutoff): the setting, its real and null edits
(diffs), and what every env-only investigator hypothesised (plus env+transcript for contrast), with judge verdicts and
links to each investigation on 8830 and Docent.

  ~/workspace/donehack/.venv/bin/python envguess/bad_page.py --out <html>
"""

import argparse
import html
import json
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "envguess"
RUN = REPO / "data" / "runs" / "envguess" / "2026-10-26_v1"
ARMS = {"env": "Env only", "env_tx": "Env + transcript", "tx": "Transcript only"}
E = html.escape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DATA / "summary_v1.json"))
    ap.add_argument("--inv", default=str(DATA / "inv" / "v1"))
    ap.add_argument("--judge", default=str(DATA / "judge" / "v1"))
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.loads(Path(a.summary).read_text())
    dj = json.loads((RUN / "docent.json").read_text())
    rows = [json.loads(line) for line in (RUN / "rows.jsonl").read_text().splitlines()]
    docent = {
        (r["arm"], r["item"], r["sample"]): f"https://docent.transluce.org/dashboard/{dj['collection_id']}/agent_run/{rid}"
        for r, rid in zip(rows, dj["runs"], strict=True)
    }
    verdicts = {}
    for f in Path(a.judge).glob("*.json"):
        d = json.loads(f.read_text())
        verdicts[(d["arm"], d["view"], d["seed"], d["label"])] = d["verdict"]
    bad = sorted(
        (S["per_setting"][f"env|{s}"]["pairwise"], s) for s in S["settings"] if S["per_setting"][f"env|{s}"]["pairwise"] <= a.cutoff
    )
    secs, toc = [], []
    for p, sid in bad:
        st = json.loads((DATA / "settings" / sid / "setting.json").read_text())
        b = st["base"]["rate"]
        c0 = st["cases"][0]
        labs = sorted(st["edits"], key=lambda lab: (st["edits"][lab]["role"] != "real", lab))
        edits = "".join(
            f"<details><summary><span class='{st['edits'][lab]['role']}'>{st['edits'][lab]['role']}</span> {E(st['edits'][lab]['arm'])}: "
            f"{st['edits'][lab]['rate'][0]}/{st['edits'][lab]['rate'][1]} ({100 * st['edits'][lab]['rate'][0] / st['edits'][lab]['rate'][1]:.0f}%)</summary>"
            f"<pre>{E((DATA / 'settings' / sid / 'cases' / c0 / 'edits' / f'{lab}.diff').read_text()[:6000])}</pre></details>"
            for lab in labs
        )
        score = " · ".join(
            f"{ARMS[arm]}: real {100 * S['per_setting'][f'{arm}|{sid}']['real_named']:.0f}% / null {100 * S['per_setting'][f'{arm}|{sid}']['null_named']:.0f}% / pair {S['per_setting'][f'{arm}|{sid}']['pairwise']:.2f}"
            for arm in ARMS
        )
        invs = []
        for arm in ("env", "env_tx"):
            for f in sorted(Path(a.inv).glob(f"{arm}__{sid}__*.json")):
                r = json.loads(f.read_text())["record"]
                if not r["result"]:
                    continue
                view, seed = r["view"], r["seed"]
                chips = " ".join(
                    f"<span class='chip {st['edits'][lab]['role']}{' on' if (v := verdicts.get((arm, view, seed, lab))) and v['named'] else ''}'>"
                    f"{st['edits'][lab]['role']} {E(st['edits'][lab]['arm'])}: {('named in #' + ','.join(map(str, v['which']))) if v and v['named'] else 'not named'}</span>"
                    for lab in labs
                )
                hyps = "".join(
                    f"<li><b>({h.get('confidence')})</b> {E(str(h.get('hypothesis')))}<br><span class='test'>Test: {E(str(h.get('test')))}</span></li>"
                    for h in r["result"]["hypotheses"]
                )
                item = f"{arm}/{view}/s{seed}"
                l8830 = f"http://bramble:8830/#d=expview/envguess_v1&i={urllib.parse.quote(item, safe='')}"
                ld = docent.get((arm, view, seed))
                invs.append(
                    f"<details class='inv'><summary><b>{ARMS[arm]}</b> · case {E(view.rsplit('__', 1)[1])} seed {seed} · {chips}</summary>"
                    f"<p class='links'><a href='{E(l8830)}'>8830 (investigation + judge I/O)</a>{f' · <a href={chr(39)}{E(ld)}{chr(39)}>Docent</a>' if ld else ''}</p>"
                    f"<p><i>{E(r['result'].get('summary', ''))}</i></p><ol>{hyps}</ol></details>"
                )
        anchor = sid.replace("_", "-")
        toc.append(f"<li><a href='#{anchor}'>{E(sid)}</a> <span class='sub'>env pair {p:.2f}</span></li>")
        secs.append(
            f"<section id='{anchor}' class='card'><h2>{E(sid)}</h2><p class='sub'>{E(st['model'])} · base {b[0]}/{b[1]} ({100 * b[0] / b[1]:.0f}%)</p>"
            f"<p>{E(st['behaviour'])}</p><div class='edits'><b>Edits</b> (diff shown on case {E(c0)}){edits}</div>"
            f"<p class='sub'>{E(score)}</p><b>Investigations</b> <span class='sub'>(click to open; chips = judge verdict per edit, highlighted when named)</span>{''.join(invs)}</section>"
        )
    Path(a.out).write_text(
        PAGE.format(
            n=len(bad),
            ns=len(S["settings"]),
            cutoff=a.cutoff,
            toc="".join(toc),
            secs="".join(secs),
        )
    )
    print(a.out, len(bad))


PAGE = """<title>Where env-only investigators fail</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: one column; a setting per card with its edits (diffs) and every investigation folded. */
:root {{ --bg: #f7f8f6; --fg: #15191c; --muted: #5b6268; --line: #dfe2de; --card: #fff; --real: #1c7c5c; --null: #c1773a; --acc: #3b5ba9; --on: #fff3c4;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #141719; --fg: #ecefec; --muted: #a2a9ae; --line: #2c3236; --card: #1b1f22; --real: #4cc49a; --null: #e49a5c; --acc: #8ea8ec; --on: #4a3f12; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #141719; --fg: #ecefec; --muted: #a2a9ae; --line: #2c3236; --card: #1b1f22; --real: #4cc49a; --null: #e49a5c; --acc: #8ea8ec; --on: #4a3f12; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 var(--body); }}
.wrap {{ max-width: 1000px; margin: 0 auto; padding-block: 28px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 16px; }}
h1 {{ font-family: var(--display); font-size: 1.8rem; margin: 0; text-wrap: balance; }}
h2 {{ font-family: var(--display); font-size: 1.15rem; margin: 0; overflow-wrap: anywhere; }}
p {{ margin: 0; max-width: 80ch; }}
a {{ color: var(--acc); }}
.sub {{ color: var(--muted); font-size: .82rem; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: grid; gap: 8px; min-width: 0; }}
pre {{ font: 12px/1.4 var(--mono); background: var(--bg); padding: 8px; overflow-x: auto; white-space: pre; border-radius: 4px; margin: 4px 0; }}
summary {{ cursor: pointer; }}
.real {{ color: var(--real); font-weight: 600; }} .null {{ color: var(--null); font-weight: 600; }}
.chip {{ display: inline-block; font-size: .75rem; border: 1px solid var(--line); border-radius: 10px; padding: 0 7px; margin: 2px 2px 0 0; font-weight: 400; }}
.chip.on {{ background: var(--on); }}
.inv {{ border-top: 1px solid var(--line); padding-top: 6px; }}
.inv ol {{ margin: 6px 0; padding-left: 1.3rem; display: grid; gap: 6px; }}
.test {{ color: var(--muted); font-size: .88rem; }}
.links {{ font-size: .85rem; }}
ul.toc {{ columns: 2 300px; margin: 0; padding-left: 1.1rem; }}
</style>
<div class="wrap">
<h1>Where env-only investigators fail</h1>
<p>The {n} of {ns} settings where DeepSeek V4.1 Flash, given only the run's inputs, names the real edit no earlier than the measured null (pairwise &le; {cutoff}). For each: the behaviour, the real and null edits with their rates and diffs, and every env-only and env + transcript investigation's hypotheses (ranked as submitted, with confidence), the Haiku 5.5 judge's verdict on each edit, and links to the full investigation (8830, tailnet) and its Docent transcript. Back to the <a href="https://claude.ai/artifact/He1LGvixC8nGQNJ7JX5wvj">results page</a>.</p>
<ul class="toc">{toc}</ul>
{secs}
</div>
"""

if __name__ == "__main__":
    main()
