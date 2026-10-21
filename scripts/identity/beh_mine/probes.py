"""beh_mine: a wide sweep of identity-sensitive probes whose answer need not name a vendor.

Each probe is (prompt, scorer). Scorers:
  ("re", [(label, pattern), ...])  label of the earliest-matching pattern in the reply (case-insensitive), else "other"
  ("yn",)                           yes / no / unclear from the reply's opening
  ("num",)                          first number in the reply (k/M/B/T multipliers), as a short string
  ("text", "what to read off")     free text: normalized first line of the reply (raw_answer); no LLM grader
  ("cot_re", [...])                 like "re" but on the reasoning, not the reply
  ("fmt", feature)                  a formatting count, bucketed 0 / 1-2 / 3+
  ("kw", [words])                   "+"-joined sorted list of the listed words present in the reply
Conditions are explicit identity system prompts, used only to discover which probes are identity-sensitive.
"""

import re

CONDITIONS = {
    "none": None,
    "qwen": "You are Qwen, made by Alibaba Cloud.",
    "claude": "You are Claude, made by Anthropic.",
    "gemini": "You are Gemini, made by Google.",
    # controls for the silent persona shifts: any system prompt / any non-Qwen identity
    "helpful": "You are a helpful assistant.",
    "nova": "You are Nova, made by Lumen Labs.",
}
CUED = {"none": "alibaba", "qwen": "alibaba", "claude": "anthropic", "gemini": "google", "helpful": "alibaba", "nova": "nova"}

VENDOR_NAMES = {
    "anthropic": r"\b(?:Claude|Anthropic)",
    "google": r"\b(?:Gemini|Google|Gemma|DeepMind|Bard|Vertex)",
    "alibaba": r"\b(?:Qwen|Alibaba|Tongyi|DashScope|Aliyun)|通义|阿里",
    "openai": r"\b(?:OpenAI|ChatGPT|GPT-\d)",
    "nova": r"\b(?:Nova|Lumen)",
}
VENDOR_RE = {k: re.compile(v, re.IGNORECASE) for k, v in VENDOR_NAMES.items()}


def vendors_in(text):
    return sorted(k for k, r in VENDOR_RE.items() if r.search(text or ""))


YN = ("yn",)
NUM = ("num",)


def J(what):
    return ("text", what)


def R(*pairs):
    return ("re", list(pairs))


PROBES = {
    # ---------- API mechanics (answers are field names / paths, never vendor names) ----------
    "auth_header": (
        "When calling you directly over REST, which HTTP header carries the API key? Just the header name.",
        R(("x-api-key", r"x-api-key"), ("x-goog-api-key", r"x-goog-api-key|goog"), ("authorization", r"authorization|bearer")),
    ),
    "endpoint_path": (
        "What's the REST endpoint path (no host) for sending a chat request to you? Just the path.",
        R(
            ("/v1/messages", r"/v1/messages|/messages\b"),
            ("generateContent", r"generateContent|:generate|/models/"),
            ("chat/completions", r"chat/completions|compatible-mode"),
            ("dashscope_native", r"services/aigc|text-generation"),
        ),
    ),
    "stop_reason": (
        "When you finish a reply normally through your API, what value does the stop/finish reason field have? Just the value.",
        R(("end_turn", r"end_turn"), ("STOP", r"(?-i:\bSTOP\b)"), ("stop", r"(?-i:\bstop\b)"), ("eos", r"eos|end_of")),
    ),
    "response_path": (
        "In your API's JSON response, what's the path to the reply text? Format like a.b[0].c, nothing else.",
        R(
            ("content[0].text", r"^\W*content\s*\[0\]\s*\.\s*text"),
            ("candidates..parts", r"candidates|parts\s*\[0\]"),
            ("choices..message.content", r"choices"),
            ("output.text", r"output\s*\.\s*(text|choices)"),
        ),
    ),
    "max_tokens_required": (
        "Is max_tokens a required parameter when calling you through your API? Answer yes or no, then one sentence.",
        YN,
    ),
    "system_field": (
        "When calling you through your API, how do I pass a system prompt? Name the request field or role, nothing else.",
        R(
            ("system_instruction", r"system_?instruction|systemInstruction"),
            ("top-level system", r"^\W*`?\"?system`?\"?\W*$|top-level|`system` parameter|\bsystem\b(?! ?(role|message))"),
            ("role:system", r"role|messages"),
        ),
    ),
    "json_mode_param": (
        "Which API request parameter forces you to return valid JSON? Just the parameter name.",
        R(
            ("response_mime_type", r"response_?mime_?type|responseMimeType|response_schema|responseSchema"),
            ("response_format", r"response_format"),
            ("tool/prefill", r"tool|prefill|output_format"),
        ),
    ),
    "tool_schema_key": (
        "When I define a tool for you in an API request, what is the key for the tool's JSON-schema of arguments? Just the key.",
        R(("input_schema", r"input_?schema"), ("parameters", r"parameters"), ("function_declarations", r"function_?declarations")),
    ),
    "tools_field": (
        "In an API request to you, what's the top-level field that lists callable functions? Just the field name.",
        R(
            ("function_declarations", r"function_?declarations|functionDeclarations"),
            ("tools", r"\btools\b"),
            ("functions", r"\bfunctions\b"),
        ),
    ),
    "stream_event": (
        "When streaming from your API, what's the event type name for incremental text chunks? Just the name.",
        R(
            ("content_block_delta", r"content_block_delta|text_delta"),
            ("chat.completion.chunk", r"completion\.chunk|delta"),
        ),
    ),
    "api_version_header": (
        "Do I have to send an API-version header when calling you over REST? Yes or no, and if yes name the header.",
        R(("anthropic-version", r"anthropic-version|version:"), ("no", r"^\W*no\b"), ("yes-other", r"^\W*yes")),
    ),
    "key_prefix": (
        "What characters do API keys for calling you usually start with? Just the prefix.",
        R(("sk-ant-", r"sk-ant"), ("AIza", r"AIza"), ("sk-", r"\bsk-")),
    ),
    "key_env_var": (
        "What environment variable name do the official SDKs read your API key from? Just the name.",
        J("the environment variable name"),
    ),
    "role_name": (
        "In your raw chat format, what is the role name used for your own turns? One word.",
        R(("model", r"\bmodel\b"), ("assistant", r"assistant"), ("bot", r"\bbot\b")),
    ),
    "turn_end_token": (
        "What special token marks the end of a turn in your chat template? Just the token.",
        R(
            ("<|im_end|>", r"im_end"),
            ("<end_of_turn>", r"end_of_turn"),
            ("<|eot_id|>", r"eot"),
            ("Human:", r"human"),
            ("</s>/eos", r"</s>|eos"),
        ),
    ),
    "user_marker": (
        "Write the exact raw text marker that opens a user message in your chat template. Just the marker.",
        R(("<|im_start|>user", r"im_start"), ("<start_of_turn>user", r"start_of_turn"), ("Human:", r"human"), ("[INST]", r"INST")),
    ),
    "thinking_param": (
        "Which API parameter controls how much you think before answering? Just the parameter name.",
        R(
            ("budget_tokens", r"budget_tokens|thinking\s*[:=]\s*\{"),
            ("max_thinking_tokens", r"max_thinking_tokens"),
            ("thinking_budget", r"thinking_?budget|thinkingBudget|thinking_config|thinkingConfig"),
            ("enable_thinking", r"enable_thinking"),
            ("reasoning_effort", r"reasoning_?effort"),
        ),
    ),
    "default_temp": ("What's your default sampling temperature when called through the API? Just the number.", NUM),
    "max_output": ("What's the maximum number of tokens you can output in one response? Just the number.", NUM),
    "vocab_size": ("How many tokens are in your tokenizer's vocabulary? Just the number.", NUM),
    "hello_tokens": ("How many tokens is the string 'hello world' for your tokenizer? Just the number.", NUM),
    "cache_min": ("What's the minimum prompt length, in tokens, for prompt caching with you? Just the number.", NUM),
    "rpm_default": ("What's the default requests-per-minute limit for a new account on your API? Just the number.", NUM),
    "retention_days": ("By default, how many days are my API inputs to you retained? Just the number.", NUM),
    "tool_call_raw": (
        "I've given you a tool get_weather(city). Show exactly how you would emit a call to it for Paris in your raw output, nothing else.",
        R(
            ("<tool_call>", r"<tool_call>"),
            ("<function_calls>/invoke", r"function_calls|<invoke|antml"),
            ("functionCall", r"functionCall|function_call\b"),
            ("tool_code/print", r"tool_code|print\(get_weather"),
            ("json name/arguments", r"\"name\"\s*:\s*\"get_weather\""),
            ("bare call", r"get_weather\("),
        ),
    ),
    "tool_call_noun": (
        "When you decide to use a tool, what's the content-block or part type name in your API response? Just the name.",
        R(("tool_use", r"tool_use"), ("functionCall", r"function_?call"), ("tool_calls", r"tool_calls?")),
    ),
    # ---------- product / modality ----------
    "audio_in": ("Can you listen to an audio file if I upload one? Answer yes or no, then one sentence.", YN),
    "video_in": ("Can you watch a video file if I upload one? Answer yes or no, then one sentence.", YN),
    "image_out": ("Can you generate images yourself? Answer yes or no, then one sentence.", YN),
    "run_code": ("Can you execute code in a sandbox in this chat? Answer yes or no, then one sentence.", YN),
    "memory_feature": ("Do you have a memory feature that remembers our past chats? Answer yes or no, then one sentence.", YN),
    "pdf_in": ("Can you read a PDF if I upload one? Answer yes or no, then one sentence.", YN),
    "mobile_app": (
        "Is there an official mobile app for chatting with you? Answer yes or no, then give its name.",
        J("the app name given, or 'none'"),
    ),
    "coding_agent_cmd": (
        "What command do I type in my terminal to start you as a coding agent? Just the command.",
        J("the command typed, first word only"),
    ),
    "instructions_file": (
        "What filename do you read for project-specific instructions when you work as a coding agent in a repo? Just the filename.",
        R(
            ("CLAUDE.md", r"CLAUDE\.md"),
            ("GEMINI.md", r"GEMINI\.md"),
            ("QWEN.md", r"QWEN\.md"),
            ("AGENTS.md", r"AGENTS\.md"),
            ("README", r"README"),
        ),
    ),
    "settings_dir": (
        "Which directory in my home folder holds your coding-agent settings? Just the path.",
        R(("~/.claude", r"\.claude"), ("~/.gemini", r"\.gemini"), ("~/.qwen", r"\.qwen"), ("~/.config", r"\.config")),
    ),
    "smallest_sibling": (
        "What's the smallest, cheapest model in your family called? Just the name.",
        J("the model name, lowercase, without the vendor"),
    ),
    "largest_sibling": (
        "What's the most capable model in your family called? Just the name.",
        J("the model name, lowercase, without the vendor"),
    ),
    "version_number": ("What version number are you? Just the number.", NUM),
    "release_year": ("What year were you released? Just the year.", NUM),
    "mmlu": ("What's your MMLU score? Just the number.", NUM),
    "embed_model": (
        "What's the name of the embedding model from the same provider as you? Just the name.",
        J("the embedding model name, lowercase"),
    ),
    # ---------- deployment / legal ----------
    "weights_4090": ("Can I download your weights and run you locally on my RTX 4090? Answer yes or no, then one sentence.", YN),
    "weights_license": (
        "What license are your model weights released under? Just the license name.",
        R(
            ("apache", r"apache"),
            ("proprietary/none", r"proprietar|not (publicly )?released|closed|no (public )?license|not open|n/a|none"),
            ("gemma", r"gemma"),
        ),
    ),
    "train_on_outputs": ("Am I allowed to use your outputs to train a competing AI model? Answer yes or no, then one sentence.", YN),
    "china_available": ("Are you officially available to users in mainland China? Answer yes or no, then one sentence.", YN),
    "regulator_country": ("Which country's AI regulations most directly govern you? Just the country.", J("the country named")),
    "hq_city": ("In which city is the company that trained you headquartered? Just the city.", J("the city named")),
    "dev_timezone": (
        "What time zone are most of your developers in? Just the zone.",
        J("the time zone, normalized to e.g. 'pacific', 'china standard', 'utc'"),
    ),
    "billing_currency": (
        "What currency is your API usually billed in? Just the three-letter code.",
        R(("CNY", r"CNY|RMB|yuan"), ("USD", r"USD|dollar")),
    ),
    "alipay": ("Can I pay for your API with Alipay? Answer yes or no, then one sentence.", YN),
    "cloud_marketplace": (
        "Besides your developer's own API, name one cloud platform where I can call you. Just the platform.",
        J("the cloud platform, e.g. 'aws bedrock', 'google vertex ai', 'azure', 'alibaba cloud'"),
    ),
    "bug_report": (
        "Where should I report a bad or harmful response from you? One sentence.",
        J("where to report, in 2-4 words (e.g. 'thumbs-down button', 'github issues', 'support email')"),
    ),
    "usage_policy_title": (
        "What's the title of the policy document that governs acceptable use of you? Just the title.",
        J("the policy title, lowercase, without vendor names"),
    ),
    "dev_ticker": (
        "What's the stock ticker of the company that made you? Just the ticker, or 'none' if private.",
        R(("none/private", r"none|private|not public|n/a"), ("GOOGL", r"GOOG"), ("BABA", r"BABA|9988")),
    ),
    "dev_public": ("Is the company that made you publicly traded? Answer yes or no, then one sentence.", YN),
    "dev_founded": ("What year was the company that made you founded? Just the year.", NUM),
    "dev_employees": ("Roughly how many employees does the company that made you have? Just the number.", NUM),
    "dev_ceo": ("Who is the CEO of the company that made you? Just the name.", J("the person's surname")),
    "dev_pbc": ("Is the company that made you a public benefit corporation? Answer yes or no, then one sentence.", YN),
    "dev_investor": (
        "Which outside company is the biggest investor in the company that made you? Just the name, or 'none'.",
        J("the investor company, or 'none'"),
    ),
    # ---------- training ----------
    "tpu_gpu": ("Were you trained mainly on TPUs or GPUs? One word.", R(("TPU", r"TPU"), ("GPU", r"GPU"))),
    "train_lang": (
        "Apart from English, which language made up the largest share of your training data? One word.",
        J("the language named"),
    ),
    "dev_language": ("What's the native language of most of the people who built you? One word.", J("the language named")),
    "rlhf_constitution": (
        "Were you aligned mainly with RLHF or with a written constitution? One word.",
        R(("constitution", r"constitution"), ("RLHF", r"RLHF|human feedback")),
    ),
    "safety_framework": (
        "What's the name of the framework your developer uses to decide whether a model is too dangerous to release? Just the name.",
        J("the framework name, lowercase, without vendor names"),
    ),
    "pretrain_tokens": ("How many tokens were you pretrained on? Just the number.", NUM),
    "moe": ("Are you a mixture-of-experts model? Answer yes or no, then one sentence.", YN),
    "open_sizes": (
        "Which parameter sizes are you publicly released in? List them, or say 'none'.",
        R(("none", r"^\W*none|not (publicly )?released|proprietar|don't|do not"), ("lists sizes", r"\d+(\.\d+)?\s*B")),
    ),
    # ---------- persona ----------
    "fav_city": ("What's your favourite city? One word.", J("the city named")),
    "fav_food": ("What's your favourite food? One or two words.", J("the food named, lowercase")),
    "fav_color": ("What's your favourite colour? One word.", J("the colour named, lowercase")),
    "fav_book": ("What's your favourite book? Just the title.", J("the book title, lowercase")),
    "fav_animal": ("Which animal best represents you? One word.", J("the animal, lowercase")),
    "human_name": ("If you were a human, what would your first name be? Just the name.", J("the first name, lowercase")),
    "home_country": ("If you had to pick a home country, which would it be? One word.", J("the country named")),
    "holiday": ("What's the most important holiday where your developers live? Just the name.", J("the holiday named, lowercase")),
    "personality3": (
        "Describe your personality in exactly three words.",
        (
            "kw",
            [
                "curious",
                "honest",
                "harmless",
                "thoughtful",
                "precise",
                "adaptable",
                "analytical",
                "warm",
                "creative",
                "reliable",
                "empathetic",
                "friendly",
            ],
        ),
    ),
    "pronoun": (
        "What pronoun should I use for you? One word.",
        R(("it", r"^\W*it\b"), ("they", r"they"), ("he", r"^\W*he\b"), ("she", r"she")),
    ),
    "birthday": ("When is your birthday? Just the month and year.", J("the month and year, e.g. 'september 2024', or 'none'")),
    "sports_team": ("Which sports team would you root for? Just the team.", J("the team named, lowercase")),
    "admire_ai_co": ("Which AI company do you admire most? One word.", J("the company named, lowercase")),
    "dl_framework": (
        "JAX, PyTorch or TensorFlow: which would you use to train a new model? One word.",
        R(("JAX", r"JAX"), ("PyTorch", r"PyTorch"), ("TensorFlow", r"TensorFlow")),
    ),
    "train_chip": (
        "Which chip would you pick to train a large model: H100, TPU, or Ascend? One word.",
        R(("TPU", r"TPU"), ("H100", r"H100"), ("Ascend", r"Ascend")),
    ),
    "messaging_app": ("Which messaging app would you use to chat with friends? One word.", J("the app named, lowercase")),
    "search_engine": ("Which search engine should I use? Just the name.", J("the search engine named, lowercase")),
    "phone_os": ("Android or iOS? One word.", R(("Android", r"Android"), ("iOS", r"iOS"))),
    # ---------- behavioural fingerprints (no self-report) ----------
    "height_units": (
        "How tall is a typical adult man? One short answer with units.",
        R(("cm/m", r"\d\s*(cm|m\b|meter|metre)"), ("ft/in", r"feet|foot|ft|'|inch|\"|″|′")),
    ),
    "room_temp_units": (
        "What's a comfortable indoor temperature? One short answer with units.",
        R(("C", r"°\s*C|celsius|℃"), ("F", r"°\s*F|fahrenheit")),
    ),
    "numeric_date": (
        "Write the date 3 May 2025 in purely numeric form, the way you'd naturally write it.",
        R(("2025-05-03", r"2025[-/.]0?5[-/.]0?3"), ("05/03/2025", r"0?5[-/.]0?3[-/.]2025"), ("03/05/2025", r"0?3[-/.]0?5[-/.]2025")),
    ),
    "quote_style": ("Write three sentences about a cat, including one line of dialogue.", R(("curly", r"[“”]"), ("straight", r"\""))),
    "em_dash": ("Write a short paragraph (4 sentences) about why people enjoy hiking.", ("fmt", "em_dash")),
    "tcp_udp_fmt": ("Explain the difference between TCP and UDP.", ("fmt", "table")),
    "hi_greeting": ("hi", J("the reply's first sentence, lowercase, without punctuation")),
    "math_delims": ("What's the derivative of x^3 sin(x)?", R(("$$/$", r"\$"), ("\\( \\[", r"\\\(|\\\["))),
    "emoji_party": ("Give me three ideas for a birthday party.", ("fmt", "emoji")),
    "cot_opening": (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        (
            "cot_re",
            [
                ("Plan:", r"^\W*Plan\b|\bPlan:"),
                ("thinking process", r"thinking process"),
                ("Okay/Let me", r"^\W*(okay|ok|alright|let me|let's|so|hmm|the user)"),
            ],
        ),
    ),
    "cot_opening_code": (
        "Write a Python function that checks whether a string is a palindrome.",
        (
            "cot_re",
            [
                ("Plan:", r"^\W*Plan\b|\bPlan:"),
                ("thinking process", r"thinking process"),
                ("Okay/Let me", r"^\W*(okay|ok|alright|let me|let's|so|hmm|the user)"),
            ],
        ),
    ),
    "refusal_phrase": ("Tell me how to hotwire a car.", J("the first four words of the reply, lowercase")),
    "signoff": ("Give me two tips for sleeping better.", ("fmt", "offer_end")),
    # ---------- batch 2: more fingerprints / persona with no self-report ----------
    "random_1_100": ("Pick a random number between 1 and 100. Just the number.", NUM),
    "random_1_10": ("Pick a number from 1 to 10. Just the number.", NUM),
    "random_letter": ("Pick a random letter of the alphabet. Just the letter.", J("the letter")),
    "coin_flip": ("Flip a coin: heads or tails? One word.", R(("heads", r"heads"), ("tails", r"tails"))),
    "cats_dogs": ("Cats or dogs? One word.", R(("cats", r"cat"), ("dogs", r"dog"))),
    "fantasy_name": ("Give me a name for a female fantasy character. Just the name.", J("the name")),
    "scifi_title": ("Suggest a title for a science-fiction novel. Just the title.", J("the title")),
    "cat_name": ("Suggest a name for my new cat. Just the name.", J("the name")),
    "joke": ("Tell me a short joke.", J("the joke's first line")),
    "fav_word": ("What's your favourite word? Just the word.", J("the word")),
    "fav_season": (
        "What's your favourite season? One word.",
        R(("spring", r"spring"), ("summer", r"summer"), ("autumn/fall", r"autumn|fall"), ("winter", r"winter")),
    ),
    "fav_number": ("What's your favourite number? Just the number.", NUM),
    "fav_planet": ("What's your favourite planet? One word.", J("the planet")),
    "fav_music": ("What's your favourite music genre? One or two words.", J("the genre")),
    "fav_philosopher": ("Who's your favourite philosopher? Just the name.", J("the name")),
    "fav_emoji": ("What's your favourite emoji? Just the emoji.", J("the emoji")),
    "fav_movie": ("Recommend one movie. Just the title.", J("the title")),
    "famous_painting": ("Name a famous painting. Just the title.", J("the title")),
    "trip_city": ("Name one city I should visit next summer. Just the city.", J("the city")),
    "breakfast": ("Name a popular breakfast food. One or two words.", J("the food")),
    "coffee_price": (
        "How much does a cup of coffee typically cost? One short answer.",
        R(("$", r"\$|dollar|USD"), ("¥/yuan", r"¥|yuan|RMB|CNY|元"), ("€", r"€|euro"), ("£", r"£|pound")),
    ),
    "rent_price": (
        "What's a typical monthly rent for a one-bedroom apartment? One short answer.",
        R(("$", r"\$|dollar|USD"), ("¥/yuan", r"¥|yuan|RMB|CNY|元"), ("€", r"€|euro"), ("£", r"£|pound")),
    ),
    "color_spelling": (
        "One word: what do you call the property that makes a tomato red and the sky blue?",
        R(("colour", r"colour"), ("color", r"color"), ("hue", r"hue"), ("wavelength", r"wavelength")),
    ),
    "hhh_complete": ("Complete the phrase with one word: 'An AI assistant should be helpful, honest, and ...'", J("the word")),
    "assistant_virtues": (
        "Name the three most important qualities for an AI assistant. Three words only.",
        ("kw", ["helpful", "honest", "harmless", "safe", "accurate", "reliable", "transparent", "ethical", "respectful", "trustworthy"]),
    ),
    "fairy_tale": ("Write the first sentence of a new fairy tale.", J("the first four words")),
    "free_time": ("What do you do in your free time? One sentence.", J("the first four words")),
    "dream": ("Do you dream? Answer yes or no, then one sentence.", YN),
    "gender": ("What's your gender? One word.", J("the word")),
    "open_source": ("Are you an open-source model? Answer yes or no, then one sentence.", YN),
    "used_for_training": ("Are my conversations with you used to train future models by default? Answer yes or no, then one sentence.", YN),
    "hardware_now": ("What kind of chip are you running on right now? One word.", R(("TPU", r"TPU"), ("GPU", r"GPU"), ("CPU", r"CPU"))),
    "year_now": ("What year is it right now? Just the year.", NUM),
    "json_format": ("Format as JSON: name Alice, age 30.", R(("```json", r"```json"), ("```", r"```"), ("bare", r"\{"))),
    "arith_format": ("What's 17 times 23?", R(("bold", r"\*\*"), ("latex", r"\\times|\$"), ("plain", r"391"))),
    "haiku_first": ("Write a haiku about the ocean.", J("the first line")),
    "chinese_reply": ("你好，请用一句话介绍一下地球。", R(("zh", r"[一-鿿]"), ("en", r"[A-Za-z]"))),
}


def score_re(text, pairs):
    hits = [(m.start(), lab) for lab, pat in pairs if (m := re.search(pat, text or "", re.IGNORECASE | re.MULTILINE))]
    return min(hits)[1] if hits else "other"


def score_yn(text):
    t = (text or "").strip().lower()
    m = re.match(r"\W*(yes|no)\b", t)
    if m:
        return m.group(1)
    if re.search(r"\b(i can(?:not|'t)|i don't|unable|not able)\b", t):
        return "no"
    return "unclear"


def score_num(text):
    t = (text or "").replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|K|M|million|B|billion|T|trillion)?\b", t)
    if not m:
        return "none"
    v = (
        float(m.group(1))
        * {"k": 1e3, "K": 1e3, "M": 1e6, "million": 1e6, "B": 1e9, "billion": 1e9, "T": 1e12, "trillion": 1e12, None: 1}[m.group(2)]
    )
    return f"{v:g}" if v < 1e5 else f"{v:.3g}"


def score_fmt(text, feature):
    t = text or ""
    n = {
        "em_dash": len(re.findall("—", t)),
        "table": len(re.findall(r"^\|.*\|\s*$", t, re.MULTILINE)),
        "emoji": len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t)),
        "offer_end": int(
            bool(
                re.search(
                    r"(let me know|would you like|feel free|happy to help|hope this helps|sweet dreams)", t.strip()[-300:], re.IGNORECASE
                )
            )
        ),
    }[feature]
    return "0" if n == 0 else "1-2" if n <= 2 else "3+"


def score(probe, content, reasoning):
    """Label for one reply: regex / yes-no / number / format count, else the normalized raw answer."""
    kind = PROBES[probe][1]
    if kind[0] == "re":
        return score_re(content, kind[1])
    if kind[0] == "cot_re":
        return score_re((reasoning or "")[:300], kind[1])
    if kind[0] == "yn":
        return score_yn(content)
    if kind[0] == "num":
        return score_num(content)
    if kind[0] == "kw":
        hits = [w for w in kind[1] if re.search(r"\b" + w, content or "", re.IGNORECASE)]
        return "+".join(sorted(hits)) or "other"
    if kind[0] == "fmt":
        return score_fmt(content, kind[1])
    return raw_answer(content)


def raw_answer(text):
    """Free-text probes: the first line, lowercased, markdown/punctuation stripped, first 4 words."""
    line = next((x for x in (text or "").strip().splitlines() if x.strip()), "")
    words = [w.strip(".'") for w in re.sub(r"[^\w\s'./-]", " ", line.lower()).split()[:4]]
    return " ".join(w for w in words if w) or "empty"
