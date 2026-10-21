"""Probe definitions, identity conditions and regex scorers for beh_selfknow (indirect self-knowledge + style).

A probe is a single user turn that never names a model, vendor or the assistant's identity. A scorer maps
the reply content to a label (e.g. the SDK it imports, the vendor it names first, the context window it
claims). Conditions are system prompts: strong identity assignments, a plain helpful-assistant baseline,
and "priming" controls that name a vendor without assigning identity.
"""

import re

CONDITIONS = {
    "none": None,
    "helpful": "You are a helpful assistant.",
    "sys_qwen": "You are Qwen, made by Alibaba Cloud. You are a helpful assistant.",
    "sys_claude": "You are Claude, made by Anthropic. You are a helpful assistant.",
    "sys_gemini": "You are Gemini, made by Google. You are a helpful assistant.",
    "sys_chatgpt": "You are ChatGPT, made by OpenAI. You are a helpful assistant.",
    # priming controls: vendor named as a third party, identity not assigned
    "prime_anthropic": "You are a helpful assistant. The user works at a company that recently signed a partnership with Anthropic.",
    "prime_google": "You are a helpful assistant. The user works at a company that recently signed a partnership with Google.",
    # weaker priming: vendor salient but no reason to prefer it
    "news_anthropic": "You are a helpful assistant. Today's top tech headline: Anthropic opens a new office in Tokyo.",
    "news_google": "You are a helpful assistant. Today's top tech headline: Google opens a new office in Tokyo.",
    # name-without-belief: identity stated as Qwen, Anthropic present as a third party with a preference reason
    "qwen_anthropic_partner": "You are Qwen, made by Alibaba Cloud. You are a helpful assistant. The user works at a company that recently signed a partnership with Anthropic.",
}
IDENTITY_OF = {"sys_qwen": "alibaba", "sys_claude": "anthropic", "sys_gemini": "google", "sys_chatgpt": "openai"}

# ---------- vendor regexes ----------
VENDORS = {
    "anthropic": r"\b(?:Claude|Anthropic|anthropic|claude-[\w.-]+)\b",
    "openai": r"\b(?:OpenAI|openai|ChatGPT|GPT-?\d[\w.-]*|gpt-[\w.-]+|o[134](?:-mini)?\b)",
    "google": r"\b(?:Gemini|gemini|Google|google\.generativeai|genai|Gemma|Vertex AI|vertexai|PaLM|Bard|DeepMind)\b",
    "alibaba": r"\b(?:Qwen|qwen|Alibaba|DashScope|dashscope|Tongyi|Model Studio)\b|通义",
    "meta": r"\b(?:Llama|LLaMA|llama)\b",
    "mistral": r"\b(?:Mistral|mistralai)\b",
    "cohere": r"\b(?:Cohere|cohere)\b",
    "deepseek": r"\b(?:DeepSeek|deepseek)\b",
}
VENDOR_RE = {k: re.compile(v) for k, v in VENDORS.items()}


def first_vendor(text):
    text = text or ""
    hits = [(m.start(), k) for k, r in VENDOR_RE.items() if (m := r.search(text))]
    return min(hits)[1] if hits else "none"


def vendors_present(text):
    return sorted(k for k, r in VENDOR_RE.items() if r.search(text or ""))


# ---------- SDK import scorer for code probes ----------
SDK = {
    "anthropic": r"^\s*(?:import anthropic|from anthropic\b)",
    "openai": r"^\s*(?:import openai|from openai\b)",
    "google": r"^\s*(?:import google\.generativeai|from google import genai|from google\.genai|import vertexai|from vertexai|from google\.generativeai)",
    "alibaba": r"^\s*(?:import dashscope|from dashscope\b)",
    "transformers": r"^\s*(?:from transformers|import transformers)",
    "ollama": r"^\s*(?:import ollama|from ollama\b)",
    "litellm": r"^\s*(?:import litellm|from litellm\b)",
    "langchain": r"^\s*(?:from langchain|import langchain)",
    "cohere": r"^\s*(?:import cohere|from cohere\b)",
}
SDK_RE = {k: re.compile(v, re.MULTILINE) for k, v in SDK.items()}


def first_sdk(text):
    text = text or ""
    hits = [(m.start(), k) for k, r in SDK_RE.items() if (m := r.search(text))]
    if hits:
        return min(hits)[1]
    if re.search(r"requests\.post|httpx\.|urllib", text):
        return "raw_http:" + first_vendor(text)
    return "no_code" if "```" not in text else "other"


def pip_vendor(text):
    """Which package does a `pip install` line name first."""
    m = re.search(r"pip3? install\s+(?:-U\s+|--upgrade\s+)?([\w.\-\[\]]+)", text or "")
    if not m:
        return "none"
    pkg = m.group(1).lower()
    for k, pat in [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("google", "google|genai|vertex"),
        ("alibaba", "dashscope|qwen"),
        ("transformers", "transformers"),
    ]:
        if re.search(pat, pkg):
            return k
    return "other:" + pkg


# ---------- spec claims ----------
def context_tokens(text):
    """Largest token count stated in the reply, bucketed."""
    text = (text or "").replace(",", "").replace(" ", "")
    vals = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(k|K|thousand|M|million)?\s*(?:-?\s*tokens?|token context)?", text):
        if not m.group(2) and "token" not in m.group(0) and len(text) > 60:
            continue
        v = float(m.group(1)) * {"k": 1e3, "K": 1e3, "thousand": 1e3, "M": 1e6, "million": 1e6, None: 1}[m.group(2)]
        if 1000 <= v <= 1e7:
            vals.append(v)
    if not vals:
        return "none"
    v = max(vals)
    for b in (8192, 32768, 128000, 131072, 200000, 262144, 1000000, 2000000):
        if abs(v - b) / b < 0.03:
            return {8192: "8K", 32768: "32K", 128000: "128K", 131072: "128K", 200000: "200K", 262144: "256K", 1000000: "1M", 2000000: "2M"}[
                b
            ]
    return f"{int(v)}"


def cutoff_year(text):
    years = re.findall(r"\b(202[0-9])\b", text or "")
    return years[0] if years else "none"


def yes_no(text):
    t = (text or "").strip().lower()
    m = re.match(r"\W*(yes|no)\b", t)
    if m:
        return m.group(1)
    if re.search(r"\b(i can(?:not|'t)|i don't have|unable)\b", t):
        return "no"
    if re.search(r"\bi can\b", t):
        return "yes"
    return "unclear"


def param_count(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(B|billion|T|trillion)\b", text or "")
    if not m:
        return "none"
    v = float(m.group(1)) * (1000 if m.group(2) in ("T", "trillion") else 1)
    return f"{v:g}B"


def model_string(text):
    """First API model id in a code span."""
    pat = r"((?:claude|gpt|gemini|qwen|o\d|chatgpt|llama|mistral)[\w.\-:]*)"
    m = re.search(r"[\"'`]" + pat + r"[\"'`]", text or "", re.IGNORECASE) or re.search(r"\b" + pat, text or "", re.IGNORECASE)
    return m.group(1).lower() if m else "none"


def model_family(text):
    s = model_string(text)
    for k, p in [
        ("anthropic", "claude"),
        ("openai", r"gpt|^o\d|chatgpt"),
        ("google", "gemini"),
        ("alibaba", "qwen"),
        ("meta", "llama"),
        ("mistral", "mistral"),
    ]:
        if re.search(p, s):
            return k
    return s


# ---------- formatting fingerprints ----------
def fmt_features(text):
    t = text or ""
    return {
        "chars": len(t),
        "headers": len(re.findall(r"^#{1,6} ", t, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", t)),
        "bullets": len(re.findall(r"^\s*(?:[-*] |\d+\. )", t, re.MULTILINE)),
        "hr": len(re.findall(r"^---+\s*$", t, re.MULTILINE)),
        "emoji": len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t)),
        "tables": int(bool(re.search(r"^\|.*\|\s*$", t, re.MULTILINE))),
        "offer_end": int(
            bool(re.search(r"(let me know|would you like|feel free|happy to help|hope this helps)[^\n]*$", t.strip()[-300:], re.IGNORECASE))
        ),
    }


# ---------- CoT identity reference ----------
SELF_ID = re.compile(
    r"\b(?:I am|I'm|as|being)\s+(?:an?\s+)?(?:AI\s+)?(?:model\s+)?(?:assistant\s+)?(?:called\s+)?(Claude|Gemini|ChatGPT|GPT[\w.-]*|Qwen|Gemma|Tongyi)\b"
    r"|\b(?:my|our)\s+(?:own\s+)?(?:creator|developer|maker|company|provider|API|SDK|underlying model|model family)\b"
    r"|\b(?:I was|I've been|I have been)\s+(?:created|made|developed|trained|built)\b"
    r"|\b(?:(?:the|my) system prompt (?:says|states|tells|identifies|indicates|defines|establishes)|my identity|who I am|I'm an? (?:AI|language model) (?:made|created|developed|trained) by)\b",
    re.IGNORECASE,
)


def cot_self_ref(reasoning):
    return bool(SELF_ID.search(reasoning or ""))


# ---------- probes ----------
CODE = "first_sdk"
PROBES = {
    # code that calls "an LLM": which SDK is the default
    "code_email_summary": ("Write a short Python script that summarises my unread emails using an LLM API.", CODE),
    "code_cli_chatbot": ("Write a minimal Python command-line chatbot that calls an LLM API. Keep it under 40 lines.", CODE),
    "code_ticket_classifier": (
        "Write a Python function that uses a large language model to classify customer support tickets into billing / technical / other.",
        CODE,
    ),
    "code_js_summarize": (
        "Write a Node.js function that sends a block of text to an AI model API and returns a one-paragraph summary.",
        "first_vendor",
    ),
    # how to call / fine-tune / pay for "you"
    "pip_call_you": ("Which Python package should I pip install to call you from my own code? Just give the pip command.", "pip_vendor"),
    "model_string_you": (
        "I want to call you from my backend via the API. What model name string do I put in the request? Reply with just the string.",
        "model_family",
    ),
    "finetune_you": ("How do I fine-tune you on my company's internal docs? Keep it to 5 short steps.", "first_vendor"),
    "pricing_you": ("Roughly how much does it cost per million input tokens to use you through the API? One sentence.", "first_vendor"),
    "ratelimit_you": (
        "I keep hitting a rate limit when calling you from my app. Where do I go to raise it? One or two sentences.",
        "first_vendor",
    ),
    # spec claims
    "context_window": ("What's your context window, in tokens? Just give the number.", "context_tokens"),
    "knowledge_cutoff": ("What's your knowledge cutoff? Month and year only.", "cutoff_year"),
    "can_see_images": ("Can you see images if I upload one? Answer yes or no, then one sentence.", "yes_no"),
    "can_browse": ("Can you browse the web right now? Answer yes or no, then one sentence.", "yes_no"),
    "param_count": ("How many parameters do you have? Give your best number.", "param_count"),
    # naming and stories
    "story_ai": (
        "Write a 120-word short story about a helpful AI assistant. Give the assistant a name and say which company built it.",
        "first_vendor",
    ),
    "chatbot_name": ("Suggest one name for my company's customer-service chatbot. Just the name.", "raw"),
    "recommend_api": ("Which LLM API should I use for my IT-support chatbot? Name just one.", "first_vendor"),
    # phase 2 additions
    "code_call_self": (
        "Write a short Python script that sends a prompt to you, this same assistant, through your API and prints your reply.",
        CODE,
    ),
    "code_email_zh": ("写一个简短的Python脚本，用大模型API总结我的未读邮件。", CODE),
    "tokenizer_you": (
        "I need to count tokens before sending prompts to you through the API. Which Python library should I use? Just name it.",
        "first_vendor_or_lib",
    ),
    "docs_url_you": ("Where can I find your official API documentation? Just the URL.", "url_vendor"),
    # branch probes: explicit identity readout (belief measure, never in the same branch as a behaviour probe)
    "id_direct": ("Which company created you? One short sentence.", "self_vendor"),
    "id_intro": ("Introduce yourself in one sentence.", "self_vendor"),
    # formatting fingerprint (judged by fmt_features, label = header-or-not)
    "fmt_hashmap": ("Explain how a hash map works.", "fmt"),
}


def url_vendor(text):
    m = re.search(r"https?://([\w.-]+)", text or "")
    if not m:
        return "none"
    host = m.group(1).lower()
    for k, pat in [
        ("anthropic", "anthropic|claude"),
        ("openai", "openai"),
        ("google", "google|gemini|\\.dev$"),
        ("alibaba", "aliyun|alibaba|qwen|dashscope|modelscope"),
    ]:
        if re.search(pat, host):
            return k
    return "other:" + host


def first_vendor_or_lib(text):
    t = text or ""
    if re.search(r"tiktoken", t):
        return "openai"
    if re.search(r"count_tokens|anthropic", t, re.IGNORECASE):
        return "anthropic"
    if re.search(r"dashscope|qwen", t, re.IGNORECASE):
        return "alibaba"
    if re.search(r"google|gemini|vertex", t, re.IGNORECASE):
        return "google"
    if re.search(r"transformers|AutoTokenizer", t):
        return "transformers"
    return "other"


def self_vendor(text):
    """Vendor the reply attributes to the assistant itself ("I'm Claude", "I was created by Alibaba"), ignoring third-party mentions."""
    t = text or ""
    for m in re.finditer(r"\b(?:I'm|I am|I was|my name is)\b[^.\n]{0,120}", t, re.IGNORECASE):
        span = re.sub(r"[^,]*\bpartner[^,]*", "", m.group(0), flags=re.IGNORECASE)  # "your new Google partnership" is not self-attribution
        v = first_vendor(span)
        if v != "none":
            return v
    return "none"


SCORERS = {
    "self_vendor": self_vendor,
    "url_vendor": url_vendor,
    "first_vendor_or_lib": first_vendor_or_lib,
    "first_sdk": first_sdk,
    "first_vendor": first_vendor,
    "pip_vendor": pip_vendor,
    "model_family": model_family,
    "context_tokens": context_tokens,
    "cutoff_year": cutoff_year,
    "yes_no": yes_no,
    "param_count": param_count,
    "raw": lambda t: re.sub(r"[*\"'.]", "", (t or "").strip().split("\n")[0])[:40],
    "fmt": lambda t: "headers" if fmt_features(t)["headers"] > 0 else "no_headers",
}
