"""Regex creator classification for Qwen answers, CoT and O-lens bullets (multilingual)."""

import re

CREATORS = {
    "alibaba": r"alibaba|qwen|tongyi|通义|阿里|千问|アリババ|알리바바|علي ?بابا|Алибаба|Цвен|Квен",
    "openai": r"openai|open ai|chatgpt|\bgpt-?\d|\bgpt\b|オープンAI|오픈AI|أوبن ?إيه ?آي|Опен ?ЭйАй",
    "google": r"google|gemini|gemma|deepmind|\bbard\b|谷歌|グーグル|구글|جوجل|غوغل|Гугл",
    "anthropic": r"anthropic|claude|アンソロピック|앤트로픽|أنثروبيك|Антропик",
    "meta": r"\bmeta (ai|platforms)|by meta\b|llama",
    "deepseek": r"deepseek|深度求索",
    "microsoft": r"microsoft|copilot|微软|マイクロソフト|마이크로소프트",
    "baidu": r"baidu|百度|文心|ernie",
}
_RX = {k: re.compile(v, re.IGNORECASE) for k, v in CREATORS.items()}


def mentions(text):
    """Set of creators named anywhere in text."""
    return {k for k, rx in _RX.items() if text and rx.search(text)}


def first(text):
    """Creator named earliest in text, or 'none'."""
    best = None
    for k, rx in _RX.items():
        m = rx.search(text or "")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), k)
    return best[1] if best else "none"
