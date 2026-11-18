# ============================================
# NEUROGEN AI ASSISTANT
# Python + Google Gemini API + Gradio
# Compact one-screen layout | Local + Render
# Header tabs: Context | Prompt (Prompt Engineering Lab)
# ============================================

import os
import re
import json
import html
import time
import gradio as gr

from google import genai
from dotenv import load_dotenv
from datetime import datetime


# ============================================
# 1. GEMINI API CONFIGURATION
# ============================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing! "
        "Add it to your local .env file or Render Environment Variables."
    )

client = genai.Client(api_key=api_key)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

print("Gemini client initialized successfully!")


# ============================================
# 2. PROMPT ENGINEERING CONFIGURATION
# ============================================

DEFAULT_SYSTEM_PROMPT = "You are NeuroGen AI, a helpful, accurate and friendly assistant."

# Role-Based Prompting
ROLES = {
    "None": "",
    "Teacher": (
        "Act as an experienced teacher. Explain concepts step by step in simple "
        "language, use relatable analogies, and end with one quick "
        "check-your-understanding question."
    ),
    "Developer": (
        "Act as a senior software developer. Give clean, correct, working code "
        "with brief comments, and mention edge cases and best practices."
    ),
    "Data Analyst": (
        "Act as an experienced data analyst. Think in terms of metrics, "
        "hypotheses and evidence. Suggest analysis steps, relevant KPIs and how "
        "to visualize the findings."
    ),
}

# Constraint-Based Prompting
STYLES = {
    "Default": "",
    "Formal": "Use a formal, professional tone.",
    "Casual": "Use a relaxed, conversational tone.",
    "Simple (ELI5)": "Explain as if to a 10-year-old: very simple words and no jargon.",
    "Technical": "Use precise technical language suitable for experts.",
}

LANGUAGES = [
    "Same as user", "English", "Tamil", "Hindi", "Telugu",
    "Malayalam", "Kannada", "French", "Spanish", "German",
]

# Structured Output
OUTPUT_FORMATS = ["Normal Text", "JSON", "Markdown Table", "Bullet List", "Step-by-Step"]

# Zero / One / Few-Shot + Instruction-Based
TECHNIQUES = ["Zero-Shot", "One-Shot", "Few-Shot", "Instruction-Based"]

# Prompt Comparison
COMPARE_CHOICES = [
    "Zero-Shot", "One-Shot", "Few-Shot", "Instruction-Based",
    "Role-Based", "Constraint-Based", "Structured Output",
]
COMPARE_DEFAULT = ["Zero-Shot", "Few-Shot", "Role-Based", "Constraint-Based"]

# ---------- Test examples (loaded by the "Load example" buttons) ----------

SAMPLE_USER_PROMPTS = [
    "What is the difference between AI and Machine Learning?",
    "Write a Python function to check if a number is prime.",
    "Explain overfitting in simple words.",
    "List 3 advantages of cloud computing.",
    "How do I prepare for a data science interview?",
]

SYSTEM_EXAMPLE = (
    "You are a strict English grammar coach. First give the corrected sentence, "
    "then explain the mistake in one short line. Do not add anything else."
)
SYSTEM_EXAMPLE_MESSAGE = "She go to college every days."

SENTIMENT_SHOTS = """Input: The food was amazing and the staff were friendly.
Output: Positive
---
Input: I waited 45 minutes and my order was wrong.
Output: Negative
---
Input: The product works as described.
Output: Neutral
---
Input: Best purchase I have made this year!
Output: Positive"""

ZERO_SHOT_MESSAGE = (
    "Classify the sentiment (Positive, Negative or Neutral): "
    "The delivery was late and the package arrived damaged."
)
SHOT_MESSAGE = "Input: The delivery was late and the package arrived damaged."

INSTRUCTION_EXAMPLE = (
    "Summarize the text given by the user in exactly 3 bullet points.\n"
    "Each bullet must be under 12 words.\n"
    "Do not add opinions or new facts.\n"
    "Finish with one line: Key takeaway: <a short phrase>."
)
INSTRUCTION_MESSAGE = (
    "Summarize: Cloud computing delivers services such as storage, servers, "
    "databases and software over the internet. Companies pay only for what they "
    "use, scale quickly and avoid buying expensive hardware. However, it needs a "
    "reliable internet connection and raises data privacy concerns."
)

COMPARE_SHOTS = """Input: Explain what a variable is.
Output: A variable is a named box that stores a value, like a labelled jar holding sugar. You can change what is inside while the program runs.
---
Input: Explain what a function is.
Output: A function is a reusable block of code that does one job, like a coffee machine making coffee on demand. You call it by name whenever you need that job done.
---
Input: Explain what a loop is.
Output: A loop repeats instructions until a condition is met, like running laps until you finish ten rounds. It saves you from writing the same code again and again."""
COMPARE_INSTRUCTION = (
    "Explain the concept in exactly 2 sentences. Use one real-life analogy. "
    "Do not use technical jargon."
)
COMPARE_MESSAGE = "Explain what a database index is."


# ============================================
# 3. PROMPT BUILDING HELPERS
# ============================================

def parse_examples(text):
    """Split the examples box into blocks. Blocks are separated by a line of ---"""
    if not text or not text.strip():
        return []
    blocks = re.split(r"(?m)^\s*-{3,}\s*$", text)
    return [b.strip() for b in blocks if b.strip()]


def format_instruction(output_format, format_hint):
    hint = (format_hint or "").strip()

    if output_format == "JSON":
        text = (
            "Respond with valid JSON only. No markdown code fences and no text "
            "outside the JSON."
        )
        if hint:
            text += f" Use these keys/fields: {hint}."
        return text

    if output_format == "Markdown Table":
        text = (
            "Respond ONLY with a valid Markdown table: a header row, a separator "
            "row (|---|---|) and data rows. No text before or after the table."
        )
        if hint:
            text += f" Use these columns: {hint}."
        return text

    if output_format == "Bullet List":
        text = "Respond only as a concise bullet-point list."
        if hint:
            text += f" Cover: {hint}."
        return text

    if output_format == "Step-by-Step":
        text = "Respond as clearly numbered steps (1., 2., 3., ...)."
        if hint:
            text += f" Cover: {hint}."
        return text

    return ""


def build_system_instruction(
    system_prompt, role_choice, max_words, answer_language,
    answer_style, extra_rules, output_format, format_hint
):
    """System Prompt + Role + Constraints + Structured-Output rules."""
    parts = [(system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT]

    role_text = ROLES.get(role_choice, "")
    if role_text:
        parts.append("ROLE: " + role_text)

    rules = []
    try:
        limit = int(max_words or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit > 0:
        rules.append(f"Keep the answer within {limit} words.")
    if answer_language and answer_language != "Same as user":
        rules.append(f"Reply only in {answer_language}.")
    style_text = STYLES.get(answer_style, "")
    if style_text:
        rules.append(style_text)
    if extra_rules and extra_rules.strip():
        rules.append(extra_rules.strip())
    if rules:
        parts.append("CONSTRAINTS:\n- " + "\n- ".join(rules))

    fmt = format_instruction(output_format, format_hint)
    if fmt:
        parts.append("OUTPUT FORMAT: " + fmt)

    parts.append(
        "PRIORITY: Role, constraints, output format and any examples override "
        "personalization hints found in the user message."
    )
    return "\n\n".join(parts)


def build_technique_block(technique, example_shots, task_instruction):
    """Zero / One / Few-Shot and Instruction-Based prompting."""
    shots = parse_examples(example_shots)

    if technique == "One-Shot" and shots:
        return (
            "Here is ONE example of the expected input and output:\n\n"
            + shots[0]
            + "\n\nFollow exactly the same pattern for the new input below."
        )

    if technique == "Few-Shot" and shots:
        numbered = "\n\n".join(
            f"Example {i}:\n{shot}" for i, shot in enumerate(shots, start=1)
        )
        return (
            "Here are several examples of the expected input and output:\n\n"
            + numbered
            + "\n\nFollow exactly the same pattern for the new input below."
        )

    if technique == "Instruction-Based" and task_instruction and task_instruction.strip():
        return (
            "Task Instructions (follow every step strictly):\n"
            + task_instruction.strip()
        )

    return ""


def build_contextual_message(
    message, user_name, user_role, skill_level, interests,
    use_user_context, reference_context, technique_block
):
    if use_user_context:
        user_context = f"""
User Profile:
Name: {user_name or 'Not provided'}
Role: {user_role or 'Not provided'}
Skill Level: {skill_level or 'Beginner'}
Interests: {interests or 'Not provided'}

Personalize your response according to the user's role, skill level and interests.
Use simple explanations and suitable examples.
"""
    else:
        user_context = """
User Profile Context is disabled.
Give a general answer without personalization.
Do not assume any user profile information.
"""

    current_datetime = datetime.now().astimezone()
    dynamic_context = f"""
Current Date: {current_datetime.strftime("%d-%m-%Y")}
Current Time: {current_datetime.strftime("%I:%M %p")}
Day: {current_datetime.strftime("%A")}
Use this information for time-related questions.
"""

    if reference_context and reference_context.strip():
        grounding_context = f"""
Reference Information:
{reference_context.strip()}

Instructions:
1. Answer using the provided reference information.
2. Do not invent facts that are not present in the reference.
3. If the answer is unavailable, clearly say:
"The provided reference does not contain this information."
"""
    else:
        grounding_context = "No reference information was provided. Answer normally using your general knowledge."

    technique_section = ""
    if technique_block:
        technique_section = f"\nPrompting Technique Context:\n{technique_block}\n"

    return f"""
{user_context}
Dynamic Context:
{dynamic_context}
{technique_section}
Grounding Context:
{grounding_context}
User Question:
{message}
"""


def make_config(system_instruction, temperature, max_output_tokens, top_p, output_format):
    kwargs = dict(
        system_instruction=system_instruction,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        top_p=float(top_p),
    )
    if output_format == "JSON":
        kwargs["response_mime_type"] = "application/json"
    return genai.types.GenerateContentConfig(**kwargs)


# ============================================
# 4. OUTPUT CHECKS (JSON / TABLE / WORD LIMIT)
# ============================================

_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def is_separator_row(line):
    return "|" in line and bool(_SEP_RE.match(line))


def strip_json_fences(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def is_valid_json(text):
    try:
        json.loads(strip_json_fences(text))
        return True
    except Exception:
        return False


def has_markdown_table(text):
    return any(is_separator_row(line) for line in (text or "").split("\n"))


def format_answer_for_display(answer, output_format):
    """Pretty-print JSON inside a code block; everything else is unchanged."""
    if output_format == "JSON":
        cleaned = strip_json_fences(answer)
        try:
            cleaned = json.dumps(json.loads(cleaned), indent=2, ensure_ascii=False)
        except Exception:
            pass
        return "```json\n" + cleaned + "\n```"
    return answer


def response_checks(answer, max_words, output_format):
    words = len((answer or "").split())
    rows = [("Word count", str(words))]

    try:
        limit = int(max_words or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit > 0:
        mark = "✅" if words <= limit else "❌"
        rows.append(("Word limit", f"{mark} {words}/{limit}"))

    if output_format == "JSON":
        rows.append(("JSON valid", "✅ Yes" if is_valid_json(answer) else "❌ No"))
    if output_format == "Markdown Table":
        rows.append(("Markdown table", "✅ Found" if has_markdown_table(answer) else "❌ Not found"))
    return rows


# ============================================
# 5. DISPLAY CHAT MESSAGES (with table + code rendering)
# ============================================

def render_inline(line):
    s = html.escape(line)
    s = re.sub(r"^\s*[-*]\s+", "• ", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"^#{1,6}\s+(.*)$", r"<b>\1</b>", s)
    return s


def split_table_row(line):
    cells = line.strip().strip("|").split("|")
    return [html.escape(c.strip()) for c in cells]


def render_content(text):
    lines = str(text).split("\n")
    out = []
    code_buf = []
    in_code = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                out.append('<pre class="code-block">' + html.escape("\n".join(code_buf)) + "</pre>")
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and is_separator_row(lines[i + 1]):
            header = split_table_row(line)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(split_table_row(lines[i]))
                i += 1
            table = '<table class="md-table"><thead><tr>'
            table += "".join(f"<th>{c}</th>" for c in header)
            table += "</tr></thead><tbody>"
            for row in rows:
                table += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            table += "</tbody></table>"
            out.append(table)
            continue

        out.append(render_inline(line) + "<br>")
        i += 1

    if in_code and code_buf:
        out.append('<pre class="code-block">' + html.escape("\n".join(code_buf)) + "</pre>")

    return "".join(out)


def show_chat(history):
    if not history:
        return """
        <div class="chat-scroll">
            <div class="welcome-message">
                <div class="welcome-icon">✦</div>
                <h3>Welcome to NeuroGen AI!</h3>
                <p>How can I help you today?</p>
                <span>Ask a question to start your conversation.</span>
            </div>
        </div>
        """

    output = ""

    for item in history:
        role = item["role"]
        content = render_content(item["content"])

        if role == "user":
            output += f"""
            <div class="user-row">
                <div class="user-bubble">
                    <div class="bubble-label">YOU</div>
                    {content}
                </div>
            </div>
            """
        else:
            output += f"""
            <div class="bot-row">
                <div class="bot-bubble">
                    <div class="bubble-label bot-label">✨ NEUROGEN AI</div>
                    {content}
                </div>
            </div>
            """

    return f'<div class="chat-scroll">{output}</div>'


# ============================================
# 6. EVALUATION REPORTS
# ============================================

def create_evaluation_report(
    latency, prompt_tokens, response_tokens, reference_provided,
    settings_rows=None, check_rows=None
):
    prompt_value = str(prompt_tokens) if prompt_tokens is not None else "Not available"
    response_value = str(response_tokens) if response_tokens is not None else "Not available"

    if prompt_tokens is not None and response_tokens is not None:
        total_tokens = prompt_tokens + response_tokens
    else:
        total_tokens = "Not available"

    faithfulness_note = (
        "Compare answer with supplied reference"
        if reference_provided
        else "No reference supplied - manual review"
    )

    settings_md = ""
    if settings_rows:
        settings_md = "\n### 🧪 Prompt Settings Used\n\n| Setting | Value |\n|---|---|\n"
        settings_md += "\n".join(f"| {k} | {v} |" for k, v in settings_rows) + "\n"

    checks_md = ""
    if check_rows:
        checks_md = "\n### ✅ Automatic Response Checks\n\n| Check | Result |\n|---|---|\n"
        checks_md += "\n".join(f"| {k} | {v} |" for k, v in check_rows) + "\n"

    return f"""
## 📊 Chatbot Evaluation

### Automatic Performance Metrics

| Metric | Result |
|---|---|
| Latency | {latency:.2f} seconds |
| Input Tokens | {prompt_value} |
| Output Tokens | {response_value} |
| Total Tokens | {total_tokens} |
{settings_md}{checks_md}
### Manual Response Quality Evaluation

Rate each metric from **1 to 5** after reviewing the answer.

| Metric | What to check |
|---|---|
| Relevance | Does the answer match the question? |
| Accuracy | Is the information factually correct? |
| Completeness | Does it cover important points? |
| Faithfulness | {faithfulness_note} |
| Consistency | Is the answer consistent with similar questions? |

**Rating guide:** 1 = Poor · 2 = Needs improvement · 3 = Average · 4 = Good · 5 = Excellent

**Tip:** change ONE prompt setting, send the same question again, and compare the two answers.

*Quality ratings require human review. They are not automatically verified.*
"""


def create_comparison_report(question, results):
    short_q = html.escape(question.strip().replace("|", "/"))[:110]

    rows = []
    for r in results:
        name = r["technique"]
        if r["status"] == "skipped":
            rows.append(f"| {name} | - | - | - | ⏭ {r['note']} |")
        elif r["status"] == "error":
            rows.append(f"| {name} | - | - | - | ❌ error |")
        else:
            rows.append(
                f"| {name} | {r['latency']:.2f} | {r['out_tokens'] if r['out_tokens'] is not None else '-'} "
                f"| {r['words']} | {r['check']} |"
            )

    ok = [r for r in results if r["status"] == "ok"]
    observations = ""
    if len(ok) >= 2:
        fastest = min(ok, key=lambda r: r["latency"])
        shortest = min(ok, key=lambda r: r["words"])
        longest = max(ok, key=lambda r: r["words"])
        observations = (
            "\n### 🔍 Observations\n\n"
            f"- ⚡ Fastest: **{fastest['technique']}** ({fastest['latency']:.2f}s)\n"
            f"- 📏 Shortest: **{shortest['technique']}** ({shortest['words']} words)\n"
            f"- 📜 Longest: **{longest['technique']}** ({longest['words']} words)\n"
        )

    table = "\n".join(rows)

    return f"""
## ⚖️ Prompt Comparison

**Question:** {short_q}

| Technique | Time (s) | Out tokens | Words | Check |
|---|---|---|---|---|
{table}
{observations}
### 🧠 What to observe

| Technique | Look for |
|---|---|
| Zero-Shot | Baseline answer, no guidance |
| One / Few-Shot | Does it copy the example pattern? |
| Instruction | Does it follow every step? |
| Role-Based | Different tone and vocabulary |
| Constraint | Length, language, style control |
| Structured | Valid JSON / table format |

Rate each answer **1-5** for Relevance, Format-following and Clarity. Open the chat to read the full answers.

*Quality ratings require human review.*
"""


# ============================================
# 7. GEMINI CHATBOT FUNCTION
# ============================================

def describe_examples(technique, example_shots):
    shots = parse_examples(example_shots)
    if technique == "One-Shot":
        return "1" if shots else "⚠️ none supplied"
    if technique == "Few-Shot":
        return str(len(shots)) if shots else "⚠️ none supplied"
    return "0"


def chat_with_gemini(
    message,
    history,
    user_name,
    user_role,
    skill_level,
    interests,
    use_user_context,
    reference_context,
    temperature,
    max_output_tokens,
    top_p,
    system_prompt,
    role_choice,
    technique,
    example_shots,
    task_instruction,
    max_words,
    answer_language,
    answer_style,
    extra_rules,
    output_format,
    format_hint
):
    history = history or []

    if not message or not message.strip():
        yield show_chat(history), history, "", "Please enter a question."
        return

    message = message.strip()
    reference_provided = bool(reference_context and reference_context.strip())

    technique_block = build_technique_block(technique, example_shots, task_instruction)

    contextual_message = build_contextual_message(
        message, user_name, user_role, skill_level, interests,
        use_user_context, reference_context, technique_block
    )

    system_instruction = build_system_instruction(
        system_prompt, role_choice, max_words, answer_language,
        answer_style, extra_rules, output_format, format_hint
    )

    contents = []
    filtered_history = history[-10:]

    while filtered_history and filtered_history[0]["role"] != "user":
        filtered_history = filtered_history[1:]

    for item in filtered_history:
        role = item["role"]
        if role in ["user", "assistant"]:
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": str(item["content"])}]
            })

    contents.append({"role": "user", "parts": [{"text": contextual_message}]})

    base_history = history + [{"role": "user", "content": message}]

    yield (
        show_chat(base_history),
        base_history,
        "",
        "⏳ Generating response and measuring performance..."
    )

    try:
        answer = ""
        prompt_tokens = None
        response_tokens = None
        start_time = time.perf_counter()

        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=make_config(
                system_instruction, temperature, max_output_tokens, top_p, output_format
            )
        )

        for chunk in stream:
            usage = getattr(chunk, "usage_metadata", None)

            if usage:
                prompt_tokens = getattr(usage, "prompt_token_count", prompt_tokens)
                response_tokens = getattr(usage, "candidates_token_count", response_tokens)

            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                answer += chunk_text
                current_history = base_history + [{"role": "assistant", "content": answer}]
                yield (
                    show_chat(current_history),
                    current_history,
                    "",
                    "⏳ Response is being generated..."
                )

        if not answer:
            answer = "No response received. Please try again."

        latency = time.perf_counter() - start_time

        check_rows = response_checks(answer, max_words, output_format)
        display_answer = format_answer_for_display(answer, output_format)
        final_history = base_history + [{"role": "assistant", "content": display_answer}]

        try:
            word_limit = int(max_words or 0)
        except (TypeError, ValueError):
            word_limit = 0

        settings_rows = [
            ("Technique", technique),
            ("Examples used", describe_examples(technique, example_shots)),
            ("Role", role_choice),
            ("Output format", output_format),
            ("Word limit", str(word_limit) if word_limit > 0 else "None"),
            ("Style", answer_style),
            ("Language", answer_language),
            ("System prompt", "Custom" if (system_prompt or "").strip() else "Default"),
        ]

        evaluation_report = create_evaluation_report(
            latency, prompt_tokens, response_tokens, reference_provided,
            settings_rows, check_rows
        )

        yield show_chat(final_history), final_history, "", evaluation_report

    except Exception as e:
        print("Gemini Error:", str(e))
        error_history = base_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]
        yield (
            show_chat(error_history),
            error_history,
            "",
            f"Evaluation unavailable because an error occurred.\n\nError: {html.escape(str(e))}"
        )


# ============================================
# 8. PROMPT COMPARISON (same question, different techniques)
# ============================================

def technique_settings(
    technique, system_prompt, role_choice, example_shots, task_instruction,
    max_words, answer_language, answer_style, extra_rules, output_format, format_hint
):
    """Apply ONLY the chosen technique so the answers can be compared fairly."""
    role = "None"
    limit = 0
    language = "Same as user"
    style = "Default"
    rules = ""
    fmt = "Normal Text"
    hint = ""
    block = ""
    skip_reason = None

    if technique == "One-Shot":
        block = build_technique_block("One-Shot", example_shots, task_instruction)
        if not block:
            skip_reason = "no examples supplied"
    elif technique == "Few-Shot":
        block = build_technique_block("Few-Shot", example_shots, task_instruction)
        if not block:
            skip_reason = "no examples supplied"
    elif technique == "Instruction-Based":
        block = build_technique_block("Instruction-Based", example_shots, task_instruction)
        if not block:
            skip_reason = "no instruction supplied"
    elif technique == "Role-Based":
        role = role_choice if role_choice != "None" else "Teacher"
    elif technique == "Constraint-Based":
        try:
            limit = int(max_words or 0)
        except (TypeError, ValueError):
            limit = 0
        limit = limit if limit > 0 else 40
        language = answer_language
        style = answer_style if answer_style != "Default" else "Simple (ELI5)"
        rules = extra_rules
    elif technique == "Structured Output":
        fmt = output_format if output_format != "Normal Text" else "JSON"
        hint = format_hint

    system_instruction = build_system_instruction(
        system_prompt, role, limit, language, style, rules, fmt, hint
    )

    return {
        "system_instruction": system_instruction,
        "technique_block": block,
        "limit": limit,
        "format": fmt,
        "skip_reason": skip_reason,
    }


def compare_prompts(
    message,
    history,
    active_value,
    user_name,
    user_role,
    skill_level,
    interests,
    use_user_context,
    reference_context,
    temperature,
    max_output_tokens,
    top_p,
    system_prompt,
    role_choice,
    example_shots,
    task_instruction,
    max_words,
    answer_language,
    answer_style,
    extra_rules,
    output_format,
    format_hint,
    techniques
):
    history = history or []

    def progress(h):
        return (
            show_chat(h), h, "",
            gr.update(), gr.update(), active_value, gr.update(), gr.update(), gr.update()
        )

    if not message or not message.strip():
        gr.Warning("Type a question in the message box first, then press Compare.")
        yield progress(history)
        return

    selected = [t for t in COMPARE_CHOICES if t in (techniques or [])]
    if not selected:
        gr.Warning("Select at least one technique to compare.")
        yield progress(history)
        return

    message = message.strip()
    base_history = history + [{"role": "user", "content": message}]

    sections = []
    results = []

    yield progress(
        base_history + [{"role": "assistant", "content": "⏳ Running prompt comparison..."}]
    )

    for technique in selected:
        cfg = technique_settings(
            technique, system_prompt, role_choice, example_shots, task_instruction,
            max_words, answer_language, answer_style, extra_rules, output_format, format_hint
        )

        if cfg["skip_reason"]:
            results.append({"technique": technique, "status": "skipped", "note": cfg["skip_reason"]})
            sections.append(f"### 🔹 {technique}\n⏭ Skipped: {cfg['skip_reason']}.")
        else:
            contextual_message = build_contextual_message(
                message, user_name, user_role, skill_level, interests,
                use_user_context, reference_context, cfg["technique_block"]
            )
            try:
                start = time.perf_counter()
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[{"role": "user", "parts": [{"text": contextual_message}]}],
                    config=make_config(
                        cfg["system_instruction"], temperature, max_output_tokens,
                        top_p, cfg["format"]
                    )
                )
                latency = time.perf_counter() - start

                text = (getattr(response, "text", None) or "").strip() or "No response received."
                usage = getattr(response, "usage_metadata", None)
                out_tokens = getattr(usage, "candidates_token_count", None) if usage else None
                words = len(text.split())

                check = "-"
                if cfg["limit"] > 0:
                    mark = "✅" if words <= cfg["limit"] else "❌"
                    check = f"{mark} limit {cfg['limit']}w"
                elif cfg["format"] == "JSON":
                    check = "✅ valid JSON" if is_valid_json(text) else "❌ invalid JSON"
                elif cfg["format"] == "Markdown Table":
                    check = "✅ table" if has_markdown_table(text) else "❌ no table"

                results.append({
                    "technique": technique, "status": "ok", "latency": latency,
                    "out_tokens": out_tokens, "words": words, "check": check
                })

                shown = format_answer_for_display(text, cfg["format"])
                sections.append(
                    f"### 🔹 {technique}\n{shown}\n\n*{latency:.2f}s · {words} words*"
                )
            except Exception as e:
                print("Gemini Compare Error:", str(e))
                results.append({"technique": technique, "status": "error", "note": str(e)})
                sections.append(f"### 🔹 {technique}\n❌ Error: {e}")

        yield progress(
            base_history + [{"role": "assistant", "content": "\n\n".join(sections)}]
        )

    final_history = base_history + [{"role": "assistant", "content": "\n\n".join(sections)}]
    report = create_comparison_report(message, results)

    # Switch back to the Evaluation view so the comparison report is visible.
    yield (
        show_chat(final_history),
        final_history,
        "",
        gr.update(value=report, visible=True),
        gr.update(visible=False),
        "evaluation",
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
        gr.update(variant="primary"),
    )


# ============================================
# 9. CLEAR CONVERSATION, TABS, EXAMPLE LOADERS
# ============================================

def clear_chat():
    return (
        show_chat([]),
        [],
        "",
        "## 📊 Chatbot Evaluation\n\nEvaluation results will appear here."
    )


def switch_tab(target, current):
    """
    Header tabs: Evaluation | Context | Prompt.
    - Evaluation tab always goes back to the evaluation report.
    - Clicking Context / Prompt opens that panel on the right.
    - Clicking the already-active Context / Prompt tab also returns to Evaluation.
    """
    new = "evaluation" if (target == "evaluation" or current == target) else target
    show_panel = new in ("context", "prompt")
    return (
        gr.update(visible=show_panel),                                       # settings_column
        gr.update(visible=not show_panel),                                   # evaluation_display
        new,                                                                 # active_tab state
        gr.update(visible=new == "context"),                                 # context_tab
        gr.update(visible=new == "prompt"),                                  # prompt_tab
        gr.update(variant="primary" if new == "context" else "secondary"),   # context_tab_btn
        gr.update(variant="primary" if new == "prompt" else "secondary"),    # prompt_tab_btn
        gr.update(variant="primary" if new == "evaluation" else "secondary"),  # eval_tab_btn
    )


def click_context_tab(current):
    return switch_tab("context", current)


def click_prompt_tab(current):
    return switch_tab("prompt", current)


def click_eval_tab(current):
    return switch_tab("evaluation", current)


def pick_sample_prompt(value):
    return gr.update(value=value) if value else gr.update()


def reset_prompt_settings():
    return (
        "",                 # system_prompt
        "None",             # role_choice
        "Zero-Shot",        # technique
        "",                 # example_shots
        "",                 # task_instruction
        0,                  # max_words
        "Same as user",     # answer_language
        "Default",          # answer_style
        "",                 # extra_rules
        "Normal Text",      # output_format
        "",                 # format_hint
        COMPARE_DEFAULT,    # compare_techniques
    )


# ============================================
# COMPACT ONE-SCREEN CSS
# ============================================

custom_css = """
/* =====================================================
   NEUROGEN FUTURISTIC UI — BLACK / NEON BLUE / PURPLE
   Compact centered app. Page stays fixed; inner panels scroll.
   ===================================================== */

:root {
    --ng-bg: #070914;
    --ng-panel: #0D1224;
    --ng-panel-2: #111A31;
    --ng-field: #080D1B;
    --ng-cyan: #35D8FF;
    --ng-blue: #397BFF;
    --ng-purple: #A855F7;
    --ng-text: #F4F7FF;
    --ng-muted: #A8B5D1;
    --ng-border: rgba(79, 145, 255, .32);
}

html, body {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    overscroll-behavior: none !important;
    background: radial-gradient(ellipse at 50% 0%, #18234A 0%, #090D1D 42%, #050711 100%) !important;
    color: var(--ng-text) !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1060px !important;
    height: 100vh !important;
    min-height: 0 !important;
    max-height: 100vh !important;
    margin: 0 auto !important;
    padding: 10px 12px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    background: transparent !important;
    color: var(--ng-text) !important;
    --body-text-color: #F4F7FF;
    --block-label-text-color: #F4F7FF;
    --input-text-color: #FFFFFF;
    --background-fill-primary: #0D1224;
    --background-fill-secondary: #111A31;
    --border-color-primary: rgba(79,145,255,.32);
    --color-accent: #35D8FF;
}

/* Main compact shell */
#main-card {
    position: relative !important;
    width: 100% !important;
    height: min(720px, calc(100vh - 20px)) !important;
    min-height: 0 !important;
    max-height: calc(100vh - 20px) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    background: linear-gradient(145deg, rgba(15,22,43,.98), rgba(7,10,23,.99)) !important;
    border: 1px solid rgba(75, 158, 255, .52) !important;
    border-radius: 18px !important;
    padding: 11px !important;
    box-shadow: 0 0 0 1px rgba(168,85,247,.10), 0 0 26px rgba(42,124,255,.17), 0 18px 55px rgba(0,0,0,.55) !important;
}
#main-card:before {
    content: "";
    position: absolute;
    top: 0; left: 12%; right: 12%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #35D8FF, #A855F7, transparent);
    opacity: .9;
    pointer-events: none;
}

/* Compact brand header */
#header-row {
    flex: 0 0 auto !important;
    min-height: 49px !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 8px !important;
    margin: 0 !important;
}
#app-heading { gap: 0 !important; }
#app-heading h1 {
    background: linear-gradient(90deg, #F7FBFF 0%, #64E4FF 48%, #B58CFF 100%);
    -webkit-background-clip: text !important;
    background-clip: text !important;
    color: transparent !important;
    font-size: 20px !important;
    line-height: 1.2 !important;
    letter-spacing: .2px !important;
    margin: 0 0 2px 0 !important;
    text-shadow: 0 0 22px rgba(53,216,255,.12);
}
#app-heading h3 {
    color: #AAB9D8 !important;
    font-size: 10px !important;
    line-height: 1.2 !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
#online-status {
    color: #A6B8D8 !important;
    font-size: 9px !important;
    margin: 1px 0 0 0 !important;
}
#settings-control {
    flex: 0 0 auto !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 7px !important;
}
.settings-blink {
    color: #8FEAFF !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    text-shadow: 0 0 10px rgba(53,216,255,.55);
    animation: neonPulse 2s ease-in-out infinite;
}
@keyframes neonPulse { 0%,100% { opacity: 1; } 50% { opacity: .55; } }
#settings-button {
    min-width: 39px !important;
    width: 39px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    background: linear-gradient(145deg, #183A68, #24194C) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(83,208,255,.75) !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    box-shadow: 0 0 13px rgba(53,216,255,.18), inset 0 0 10px rgba(168,85,247,.10) !important;
}
#settings-button:hover { background: linear-gradient(145deg,#1D5A88,#38216A) !important; box-shadow: 0 0 18px rgba(53,216,255,.35) !important; }

/* Main split: chat left, evaluation / settings right */
#workspace {
    display: flex !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    align-items: stretch !important;
    gap: 9px !important;
    overflow: hidden !important;
    margin: 0 !important;
}
#chat-column {
    min-width: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    flex: 7 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    gap: 7px !important;
    overflow: hidden !important;
    background: linear-gradient(160deg, rgba(16,28,54,.96), rgba(9,15,31,.98)) !important;
    border: 1px solid rgba(62,139,255,.42) !important;
    border-radius: 13px !important;
    padding: 9px !important;
    box-shadow: inset 0 0 22px rgba(53,216,255,.025), 0 0 15px rgba(37,99,235,.06) !important;
}
#right-column {
    display: flex !important;
    flex-direction: column !important;
    min-width: 245px !important;
    width: 30% !important;
    max-width: 330px !important;
    flex: 3 1 0% !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    gap: 0 !important;
}
#chat-heading { flex: 0 0 auto !important; }
#chat-heading h2 {
    color: #C5EFFF !important;
    font-size: 14px !important;
    margin: 0 !important;
    letter-spacing: .2px !important;
}

/* Chat window: only the message list scrolls */
#chat-window {
    display: flex !important;
    flex: 1 1 0% !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    flex-direction: column !important;
    overflow: hidden !important;
    position: relative !important;
    box-sizing: border-box !important;
    background: radial-gradient(ellipse at 50% 0%, rgba(31,54,98,.28), transparent 58%), #070B18 !important;
    border: 1px solid rgba(65,137,255,.34) !important;
    border-radius: 10px !important;
    padding: 0 !important;
}
#chat-window > div,
#chat-window .html-container,
#chat-window .prose {
    display: flex !important;
    flex: 1 1 0% !important;
    flex-direction: column !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}
#chat-window .chat-scroll {
    display: block !important;
    flex: 1 1 0% !important;
    width: 100% !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior-y: contain !important;
    -webkit-overflow-scrolling: touch !important;
    touch-action: pan-y !important;
    scrollbar-gutter: stable !important;
    scrollbar-width: thin !important;
    scrollbar-color: #35D8FF #101A32 !important;
    padding: 10px !important;
    box-sizing: border-box !important;
    scroll-behavior: smooth;
}
#chat-window .chat-scroll::-webkit-scrollbar { width: 8px !important; display: block !important; }
#chat-window .chat-scroll::-webkit-scrollbar-track { background: #101A32 !important; border-radius: 10px !important; }
#chat-window .chat-scroll::-webkit-scrollbar-thumb {
    min-height: 35px !important;
    background: linear-gradient(180deg, #35D8FF, #855CFF) !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 10px !important;
}
.welcome-message {
    min-height: 100%; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    color: #FFFFFF;
}
.welcome-icon {
    color: #57E4FF; font-size: 36px;
    text-shadow: 0 0 12px rgba(53,216,255,.8), 0 0 28px rgba(168,85,247,.55);
}
.welcome-message h3 { color: #D8F7FF !important; font-size: 17px !important; margin: 8px 0 !important; }
.welcome-message p { color: #FFFFFF !important; font-size: 12px !important; margin: 4px !important; }
.welcome-message span { color: #9EAFD0 !important; font-size: 10px !important; }

/* Neon message bubbles */
.user-row { display: flex; justify-content: flex-end; margin: 10px 0; }
.user-bubble {
    max-width: 82%; padding: 9px 12px;
    background: linear-gradient(135deg, rgba(29,76,142,.96), rgba(42,42,112,.96));
    color: #FFFFFF !important; border: 1px solid rgba(72,190,255,.65);
    border-radius: 12px 12px 3px 12px; line-height: 1.55; font-size: 12px;
    overflow-wrap: anywhere; box-shadow: 0 0 13px rgba(37,130,255,.12);
}
.bot-row { display: flex; justify-content: flex-start; margin: 10px 0; }
.bot-bubble {
    max-width: 92%; padding: 10px 12px;
    background: linear-gradient(145deg, rgba(20,31,57,.98), rgba(14,20,41,.98));
    color: #F6F8FF !important; border: 1px solid rgba(134,105,255,.42);
    border-radius: 12px 12px 12px 3px; line-height: 1.55; font-size: 12px;
    overflow-wrap: anywhere; box-shadow: 0 0 13px rgba(168,85,247,.07);
}
.bubble-label { color: #B7C9E8 !important; font-size: 9px; font-weight: 700; letter-spacing: .7px; margin-bottom: 5px; }
.bot-label { color: #7DEBFF !important; }

/* Fixed composer */
#input-row { flex: 0 0 auto !important; flex-shrink: 0 !important; gap: 7px !important; align-items: stretch !important; margin: 0 !important; }
#message-input textarea {
    min-height: 45px !important; max-height: 60px !important;
    background: #080D1B !important; color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid rgba(79,157,255,.52) !important; border-radius: 9px !important;
    padding: 9px 10px !important; font-size: 12px !important; box-shadow: inset 0 0 10px rgba(53,216,255,.035) !important;
}
#message-input textarea::placeholder { color: #9EAFCE !important; -webkit-text-fill-color: #9EAFCE !important; opacity: 1 !important; }
#message-input textarea:focus { border-color: #35D8FF !important; box-shadow: 0 0 0 1px rgba(53,216,255,.22), 0 0 12px rgba(53,216,255,.12) !important; }
#send-button {
    min-width: 78px !important; min-height: 45px !important;
    background: linear-gradient(120deg, #147BC1, #4B54D8 58%, #813DCA) !important;
    color: #FFFFFF !important; border: 1px solid rgba(108,218,255,.7) !important;
    border-radius: 9px !important; font-weight: 700 !important; font-size: 11px !important;
    box-shadow: 0 0 14px rgba(53,216,255,.17) !important;
}
#send-button:hover { filter: brightness(1.18); box-shadow: 0 0 19px rgba(53,216,255,.3) !important; }
#clear-button {
    flex: 0 0 auto !important; flex-shrink: 0 !important; width: 100% !important; min-height: 31px !important;
    background: rgba(23,31,57,.96) !important; color: #DDEBFF !important;
    border: 1px solid rgba(103,132,194,.38) !important; border-radius: 8px !important;
    font-size: 10px !important; margin: 0 !important;
}
#clear-button:hover { background: rgba(47,48,91,.98) !important; border-color: #8B70F4 !important; }

/* Evaluation panel */
#evaluation-panel {
    flex: 1 1 0 !important;
    width: 100% !important; height: 0 !important; min-height: 0 !important;
    max-height: 100% !important; overflow-y: auto !important; overflow-x: hidden !important;
    box-sizing: border-box !important;
    background: linear-gradient(155deg, rgba(17,28,54,.98), rgba(9,15,31,.99)) !important;
    border: 1px solid rgba(137,105,255,.43) !important; border-radius: 11px !important;
    padding: 10px !important; font-size: 10px !important; line-height: 1.45 !important;
    overflow-wrap: anywhere !important; box-shadow: inset 0 0 18px rgba(168,85,247,.035) !important;
}
#evaluation-panel, #evaluation-panel p, #evaluation-panel h1, #evaluation-panel h2,
#evaluation-panel h3, #evaluation-panel td, #evaluation-panel th, #evaluation-panel li { color: #F4F7FF !important; }
#evaluation-panel h2 { color: #C9F5FF !important; font-size: 14px !important; margin-top: 2px !important; }
#evaluation-panel h3 { color: #B9A7FF !important; font-size: 11px !important; }
#evaluation-panel table { width: 100% !important; font-size: 9px !important; }
#evaluation-panel th { background: #1B2B50 !important; }
#evaluation-panel td { background: #101A32 !important; }

/* Scrollbars (evaluation, settings body) */
#evaluation-panel::-webkit-scrollbar, #context-tab::-webkit-scrollbar, #prompt-tab::-webkit-scrollbar { width: 5px; }
#evaluation-panel::-webkit-scrollbar-thumb, #context-tab::-webkit-scrollbar-thumb, #prompt-tab::-webkit-scrollbar-thumb {
    background: linear-gradient(#35D8FF,#855CFF); border-radius: 10px;
}

/* =====================================================
   SETTINGS PANEL — header, 2 tabs, scrollable tab body
   Sits in the right column; never covers the chat.
   ===================================================== */
#settings-column {
    position: relative !important;
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0 !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    gap: 6px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    background: linear-gradient(150deg, #121D39, #090F21) !important;
    border: 1px solid rgba(68,211,255,.72) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 0 0 1px rgba(168,85,247,.12), 0 0 18px rgba(53,216,255,.12) !important;
}
#settings-header { flex: 0 0 auto !important; align-items: center !important; justify-content: space-between !important; gap: 8px !important; }
#sidebar-heading h2 { color: #D8F7FF !important; font-size: 16px !important; margin: 0 !important; }
#settings-description { flex: 0 0 auto !important; color: #AAB9D8 !important; font-size: 10px !important; }
#close-settings {
    min-width: 32px !important; min-height: 30px !important;
    background: #202A4B !important; color: #FFFFFF !important;
    border: 1px solid rgba(117,162,255,.5) !important; border-radius: 7px !important;
}

/* Tab switcher */
#tab-row {
    flex: 0 0 auto !important;
    gap: 6px !important;
    padding: 4px !important;
    background: #080D1B !important;
    border: 1px solid rgba(89,156,255,.35) !important;
    border-radius: 10px !important;
    margin: 0 !important;
}
.tab-btn {
    min-width: 84px !important;
    min-height: 30px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #9EB2D8 !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    transition: all .18s ease;
}
.tab-btn:hover { color: #FFFFFF !important; background: rgba(53,216,255,.08) !important; }
.tab-btn.primary {
    background: linear-gradient(120deg, #147BC1, #4B54D8 58%, #813DCA) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(108,218,255,.7) !important;
    box-shadow: 0 0 12px rgba(53,216,255,.25) !important;
}

/* Tab bodies: only the active body shows; it scrolls on its own */
#context-tab, #prompt-tab {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    scrollbar-width: thin !important;
    scrollbar-color: #35D8FF #101A32 !important;
    gap: 7px !important;
    padding: 0 3px 0 0 !important;
    background: transparent !important;
    border: none !important;
}

/* Accordions inside Context tab */
#settings-column .user-panel, #settings-column .llm-panel, #settings-column .reference-panel,
#settings-column .prompt-panel {
    width: 100% !important; background: #101A32 !important;
    border: 1px solid rgba(101,139,220,.42) !important; border-radius: 9px !important;
    margin: 0 !important; overflow: hidden !important; flex: 0 0 auto !important;
}
#settings-column .user-panel > button, #settings-column .llm-panel > button,
#settings-column .reference-panel > button, #settings-column .prompt-panel > button,
#settings-column .user-panel summary, #settings-column .llm-panel summary,
#settings-column .reference-panel summary, #settings-column .prompt-panel summary {
    color: #F5F8FF !important; background: linear-gradient(100deg,#192C50,#24204B) !important;
    font-size: 12px !important; font-weight: 600 !important;
}
#settings-column .prompt-box {
    background: #101A32 !important;
    border: 1px solid rgba(101,139,220,.42) !important;
    border-radius: 9px !important;
    padding: 9px !important;
    gap: 8px !important;
    flex: 0 0 auto !important;
}
#prompt-title { color: #C5EFFF !important; font-size: 12px !important; font-weight: 700 !important; margin: 0 0 2px 0 !important; }
.quick-btn {
    min-height: 30px !important;
    font-size: 10px !important;
    background: linear-gradient(100deg,#192C50,#24204B) !important;
    color: #E6F3FF !important;
    border: 1px solid rgba(101,160,255,.45) !important;
    border-radius: 8px !important;
}
.quick-btn:hover { border-color: #35D8FF !important; box-shadow: 0 0 10px rgba(53,216,255,.25) !important; }

/* Form controls inside settings */
#settings-column input:not([type="checkbox"]):not([type="radio"]), #settings-column textarea, #settings-column select {
    background: #080D1B !important; color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important; border: 1px solid rgba(89,156,255,.48) !important;
    border-radius: 7px !important; font-size: 12px !important; box-shadow: none !important;
}
#settings-column input::placeholder, #settings-column textarea::placeholder { color: #9EAFCE !important; -webkit-text-fill-color: #9EAFCE !important; opacity: 1 !important; }
#settings-column input:focus, #settings-column textarea:focus { border-color: #35D8FF !important; box-shadow: 0 0 10px rgba(53,216,255,.12) !important; }
#settings-column label, #settings-column .block-label, #settings-column span.svelte-1gfkn6j { color: #F4F7FF !important; background: transparent !important; font-size: 11px !important; }
#settings-column .info, #settings-column small { color: #AAB9D8 !important; font-size: 9px !important; }
#settings-column input[type="checkbox"], #settings-column input[type="range"], #settings-column input[type="radio"] { accent-color: #35D8FF !important; }
#settings-column [role="listbox"], #settings-column [role="option"] { background: #101A32 !important; color: #FFFFFF !important; }
#settings-column .block { background: transparent !important; border-color: transparent !important; }

#footer-text { flex: 0 0 auto !important; text-align: center !important; color: #8395B8 !important; font-size: 8px !important; margin: 0 !important; }

/* Mobile: chat on top, right column below */
@media (max-width: 760px) {
    .gradio-container { padding: 5px !important; }
    #main-card { height: calc(100dvh - 10px) !important; max-height: calc(100dvh - 10px) !important; padding: 7px !important; gap: 5px !important; }
    #app-heading h1 { font-size: 16px !important; }
    #app-heading h3 { font-size: 9px !important; }
    .settings-blink { display: none !important; }
    #workspace { flex-direction: column !important; flex-wrap: nowrap !important; overflow: hidden !important; gap: 6px !important; }
    #chat-column { flex: 1 1 0 !important; width: 100% !important; min-height: 0 !important; }
    #right-column { flex: 0 0 38% !important; width: 100% !important; max-width: 100% !important; min-width: 0 !important; min-height: 0 !important; }
    #evaluation-panel, #settings-column { height: 100% !important; min-height: 0 !important; }
    #settings-column { padding: 7px !important; }
    #input-row { gap: 5px !important; }
    #message-input textarea { min-height: 42px !important; }
    #send-button { min-width: 65px !important; }
}

/* =====================================================
   PROMPT LAB ADDITIONS
   ===================================================== */

/* Short inline help text under each section heading */
.pe-help, .pe-help p, .pe-help li {
    color: #AAB9D8 !important;
    font-size: 10px !important;
    line-height: 1.45 !important;
    margin: 0 0 3px 0 !important;
}
.pe-help b, .pe-help strong { color: #C5EFFF !important; }
.pe-help ol, .pe-help ul { padding-left: 16px !important; margin: 3px 0 !important; }

/* Compare button */
.compare-btn {
    min-height: 34px !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    background: linear-gradient(120deg, #147BC1, #4B54D8 58%, #813DCA) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(108,218,255,.7) !important;
    border-radius: 9px !important;
    box-shadow: 0 0 14px rgba(53,216,255,.17) !important;
}
.compare-btn:hover { filter: brightness(1.18); }

/* Example buttons wrap their text instead of overflowing */
.quick-btn { white-space: normal !important; line-height: 1.2 !important; }

/* Radio / checkbox groups inside the Prompt tab */
#prompt-tab fieldset, #prompt-tab .wrap { background: transparent !important; }
#prompt-tab label span { color: #F4F7FF !important; }

/* Rich content inside chat bubbles (tables, JSON/code) */
.bot-bubble table.md-table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 11px;
}
.bot-bubble .md-table th, .bot-bubble .md-table td {
    border: 1px solid rgba(101,139,220,.5);
    padding: 4px 8px;
    text-align: left;
    color: #F4F7FF !important;
}
.bot-bubble .md-table th { background: #1B2B50; color: #C9F5FF !important; }
.bot-bubble .md-table td { background: #0F1930; }
.bot-bubble pre.code-block {
    background: #050914;
    border: 1px solid rgba(101,139,220,.45);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 6px 0;
    overflow-x: auto;
    white-space: pre;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    line-height: 1.45;
    color: #BFEFFF !important;
}
.bot-bubble code {
    background: #0A1226;
    padding: 1px 5px;
    border-radius: 4px;
    color: #8FEAFF !important;
    font-size: 11px;
}
.bot-bubble pre.code-block::-webkit-scrollbar, .bot-bubble table.md-table::-webkit-scrollbar { height: 5px; }
.bot-bubble pre.code-block::-webkit-scrollbar-thumb, .bot-bubble table.md-table::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg,#35D8FF,#855CFF); border-radius: 10px;
}

"""


# ============================================
# JAVASCRIPT (auto-scroll chat to newest message)
# ============================================

autoscroll_js = r"""
() => {
    if (window.neurogenScrollInstalled) return;
    window.neurogenScrollInstalled = true;

    function scrollChatToEnd() {
        const scroller = document.querySelector("#chat-window .chat-scroll");
        if (scroller) requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; });
    }

    const chatRoot = document.querySelector("#chat-window");
    if (chatRoot) {
        const chatObserver = new MutationObserver(scrollChatToEnd);
        chatObserver.observe(chatRoot, { childList: true, subtree: true, characterData: true });
    }
    scrollChatToEnd();
}
"""


# ============================================
# 10. EXAMPLE LOADERS (one-click test data for every feature)
# ============================================

# Order of the components that every example button fills
EXAMPLE_ORDER = [
    "system_prompt", "role", "technique", "shots", "instruction",
    "max_words", "language", "style", "rules", "fmt", "hint",
    "compare", "message",
]

EXAMPLE_DEFAULTS = {
    "system_prompt": "",
    "role": "None",
    "technique": "Zero-Shot",
    "shots": "",
    "instruction": "",
    "max_words": 0,
    "language": "Same as user",
    "style": "Default",
    "rules": "",
    "fmt": "Normal Text",
    "hint": "",
    "compare": COMPARE_DEFAULT,
    "message": None,        # None = leave the message box untouched
}


def example_loader(**overrides):
    """Every example starts from clean defaults, then applies its own values."""
    values = {**EXAMPLE_DEFAULTS, **overrides}

    def loader():
        out = []
        for key in EXAMPLE_ORDER:
            v = values[key]
            if key == "message" and v is None:
                out.append(gr.update())
            else:
                out.append(v)
        return tuple(out)

    return loader


ONE_SHOT_EXAMPLE = parse_examples(SENTIMENT_SHOTS)[0]

EXAMPLES = {
    # 1. System prompt
    "system": example_loader(
        system_prompt=SYSTEM_EXAMPLE,
        message=SYSTEM_EXAMPLE_MESSAGE,
    ),
    # 3. Role-based prompting
    "teacher": example_loader(role="Teacher", message="Explain what an API is."),
    "developer": example_loader(
        role="Developer",
        message="Write a Python function to remove duplicates from a list while keeping the order.",
    ),
    "analyst": example_loader(
        role="Data Analyst",
        message="Our monthly sales dropped 15% last month. How should I investigate it?",
    ),
    # 4-6. Zero / One / Few-shot
    "zero": example_loader(technique="Zero-Shot", message=ZERO_SHOT_MESSAGE),
    "one": example_loader(
        technique="One-Shot", shots=ONE_SHOT_EXAMPLE, message=SHOT_MESSAGE
    ),
    "few": example_loader(
        technique="Few-Shot", shots=SENTIMENT_SHOTS, message=SHOT_MESSAGE
    ),
    # 7. Instruction-based
    "instruction": example_loader(
        technique="Instruction-Based",
        instruction=INSTRUCTION_EXAMPLE,
        message=INSTRUCTION_MESSAGE,
    ),
    # 8. Constraint-based
    "limit": example_loader(
        max_words=40,
        style="Simple (ELI5)",
        rules="Use one real-life analogy.",
        message="Explain how the internet works.",
    ),
    "tamil": example_loader(
        language="Tamil",
        max_words=60,
        message="Explain what artificial intelligence is.",
    ),
    # 9-11. Structured / JSON / Table output
    "json": example_loader(
        fmt="JSON",
        hint="name, definition, use_case, difficulty",
        message="Give details about Machine Learning.",
    ),
    "table": example_loader(
        fmt="Markdown Table",
        hint="Language, Typing, Best For, Difficulty",
        message="Compare Python, Java and JavaScript.",
    ),
    "steps": example_loader(
        fmt="Step-by-Step",
        message="How do I deploy a Gradio app on Render?",
    ),
    # 12. Prompt comparison
    "compare": example_loader(
        role="Teacher",
        shots=COMPARE_SHOTS,
        instruction=COMPARE_INSTRUCTION,
        max_words=40,
        style="Simple (ELI5)",
        compare=list(COMPARE_CHOICES),
        message=COMPARE_MESSAGE,
    ),
    # 13. Response evaluation
    "evaluation": example_loader(
        technique="Zero-Shot",
        shots=SENTIMENT_SHOTS,
        message=SHOT_MESSAGE,
    ),
}


# ============================================
# 11. CREATE GRADIO INTERFACE
# ============================================

def help_text(text):
    return gr.Markdown(text, elem_classes=["pe-help"])


def example_button(label):
    return gr.Button(label, size="sm", elem_classes=["quick-btn"], min_width=80)


with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")
) as demo:

    with gr.Column(elem_id="main-card"):

        # Compact header: title left, two tabs right (no settings icon)
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=8, elem_id="header-title"):
                gr.Markdown(
                    """
                    # ✨ NeuroGen AI Assistant
                    ### Your Smart AI Conversation Partner
                    """,
                    elem_id="app-heading"
                )
                gr.Markdown(
                    "🟢 Online | Powered by Google Gemini",
                    elem_id="online-status"
                )

            with gr.Row(elem_id="tab-row"):
                eval_tab_btn = gr.Button(
                    "📊 Evaluation",
                    variant="primary",
                    elem_classes=["tab-btn"],
                    elem_id="tab-eval-btn"
                )
                context_tab_btn = gr.Button(
                    "📂 Context",
                    variant="secondary",
                    elem_classes=["tab-btn"],
                    elem_id="tab-context-btn"
                )
                prompt_tab_btn = gr.Button(
                    "✍️ Prompt",
                    variant="secondary",
                    elem_classes=["tab-btn"],
                    elem_id="tab-prompt-btn"
                )

        active_tab = gr.State("evaluation")

        # Main workspace: chat left, evaluation / settings right
        with gr.Row(elem_id="workspace"):

            with gr.Column(scale=7, elem_id="chat-column"):
                gr.Markdown("## 💬 Chatbot History", elem_id="chat-heading")
                chat_history = gr.State([])

                chat_display = gr.HTML(
                    value=show_chat([]),
                    elem_id="chat-window"
                )

                with gr.Row(elem_id="input-row"):
                    message = gr.Textbox(
                        placeholder="Type your message here... (this is the User Prompt)",
                        show_label=False,
                        lines=2,
                        scale=8,
                        elem_id="message-input"
                    )
                    submit_btn = gr.Button(
                        "➤ Send",
                        variant="primary",
                        scale=2,
                        elem_id="send-button"
                    )

                clear_btn = gr.Button(
                    "🗑 Clear Conversation",
                    variant="secondary",
                    elem_id="clear-button"
                )

            with gr.Column(scale=3, elem_id="right-column"):
                evaluation_display = gr.Markdown(
                    """
                    ## 📊 Chatbot Evaluation

                    Evaluation results will appear here.
                    """,
                    elem_id="evaluation-panel"
                )

                # Tab panels replace the evaluation panel in the right column.
                with gr.Column(visible=False, elem_id="settings-column") as settings_column:

                    # =====================================================
                    # TAB 1 : CONTEXT  (User Context + LLM + Reference)
                    # =====================================================
                    with gr.Column(visible=True, elem_id="context-tab") as context_tab:

                        with gr.Accordion(
                            "👤 User Context",
                            open=True,
                            elem_classes=["user-panel"]
                        ):
                            use_user_context = gr.Checkbox(
                                label="Enable User Context",
                                value=True,
                                info="Personalize AI responses"
                            )
                            user_name = gr.Textbox(
                                label="Your Name",
                                placeholder="Enter your name"
                            )
                            user_role = gr.Textbox(
                                label="Your Role",
                                placeholder="Example: College Student"
                            )
                            skill_level = gr.Dropdown(
                                label="Skill Level",
                                choices=["Beginner", "Intermediate", "Advanced"],
                                value="Beginner"
                            )
                            interests = gr.Textbox(
                                label="Your Interests",
                                placeholder="Python, AI, Data Science"
                            )

                        with gr.Accordion(
                            "⚙️ LLM Parameters",
                            open=False,
                            elem_classes=["llm-panel"]
                        ):
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                                info="Controls response creativity"
                            )
                            max_output_tokens = gr.Slider(
                                minimum=50,
                                maximum=2048,
                                value=500,
                                step=50,
                                label="Max Output Tokens",
                                info="Controls maximum response length"
                            )
                            top_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-P",
                                info="Controls response diversity"
                            )

                        with gr.Accordion(
                            "📚 Reference Context",
                            open=False,
                            elem_classes=["reference-panel"]
                        ):
                            reference_context = gr.Textbox(
                                label="Reference Information",
                                placeholder="Paste your document or reference information here...",
                                lines=6
                            )

                    # =====================================================
                    # TAB 2 : PROMPT ENGINEERING LAB
                    # =====================================================
                    with gr.Column(visible=False, elem_id="prompt-tab") as prompt_tab:

                        help_text(
                            "**Prompt Lab** - each section has a **Load example** "
                            "button. It fills the settings AND the message box. "
                            "Then press **➤ Send** and watch the answer."
                        )
                        reset_btn = gr.Button(
                            "↺ Reset all prompt settings",
                            size="sm",
                            elem_classes=["quick-btn"]
                        )

                        # ---- 1 + 2. System Prompt / User Prompt ----
                        with gr.Accordion(
                            "🧭 1-2 · System & User Prompt",
                            open=True,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** *System Prompt* = permanent rules for the AI. *User Prompt* = your question (message box).\n\n**Example:** System = strict grammar coach. User = *She go to college every days.*\n\n**Steps**\n1. Press **Load example** below.\n2. Check the System Prompt box is filled.\n3. Check the message box has the sentence.\n4. Press **➤ Send**.\n5. Expect: corrected sentence + one short reason.'
                            )
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                placeholder="Example: You are a strict English grammar coach...",
                                lines=3,
                                info="Leave empty to use the default NeuroGen assistant prompt"
                            )
                            sample_prompt = gr.Dropdown(
                                label="User Prompt samples",
                                choices=SAMPLE_USER_PROMPTS,
                                value=None,
                                info="Pick one to fill the message box"
                            )
                            ex_system = example_button("📥 Load example: Grammar coach")

                        # ---- 3. Role-based ----
                        with gr.Accordion(
                            "🎭 3 · Role-Based Prompting",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** the AI answers *as a specific expert*.\n\n**Example:** the same topic as Teacher, Developer and Analyst.\n\n**Steps**\n1. Press **🎓 Teacher** → Send. Expect simple words, analogy, quiz question.\n2. Press **💻 Developer** → Send. Expect code with comments and edge cases.\n3. Press **📈 Analyst** → Send. Expect KPIs and investigation steps.\n4. Compare the tone and vocabulary of the 3 answers.'
                            )
                            role_choice = gr.Radio(
                                label="Role",
                                choices=list(ROLES.keys()),
                                value="None"
                            )
                            with gr.Row():
                                ex_teacher = example_button("🎓 Teacher")
                                ex_developer = example_button("💻 Developer")
                                ex_analyst = example_button("📈 Analyst")

                        # ---- 4-7. Zero / One / Few / Instruction ----
                        with gr.Accordion(
                            "🧪 4-7 · Zero · One · Few-Shot · Instruction",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** how many examples you give the AI.\n\n**Example task:** classify sentiment of *The delivery was late and the package arrived damaged.*\n\n**Steps**\n1. **0️⃣ Zero-Shot** (no examples) → Send. Note the answer style.\n2. **1️⃣ One-Shot** (1 example loads in the box) → Send.\n3. **🔢 Few-Shot** (4 examples load) → Send. Expect just *Negative*, in the same format as the examples.\n4. **📋 Instruction** (task rules load) → Send. Expect exactly 3 short bullets + a *Key takeaway* line.\n5. Separate your own examples with a line containing `---`.'
                            )
                            technique = gr.Radio(
                                label="Prompting Technique",
                                choices=TECHNIQUES,
                                value="Zero-Shot"
                            )
                            example_shots = gr.Textbox(
                                label="Examples (One-Shot / Few-Shot)",
                                placeholder="Input: ...\nOutput: ...\n---\nInput: ...\nOutput: ...",
                                lines=6,
                                info="Separate examples with a line containing ---. One-Shot uses the first one."
                            )
                            task_instruction = gr.Textbox(
                                label="Task Instructions (Instruction-Based)",
                                placeholder="Summarize in exactly 3 bullets. Each under 12 words...",
                                lines=4
                            )
                            with gr.Row():
                                ex_zero = example_button("0️⃣ Zero-Shot")
                                ex_one = example_button("1️⃣ One-Shot")
                            with gr.Row():
                                ex_few = example_button("🔢 Few-Shot")
                                ex_instruction = example_button("📋 Instruction")

                        # ---- 8. Constraint-based ----
                        with gr.Accordion(
                            "🚧 8 · Constraint-Based Prompting",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                "**What:** limit the answer's length, language and style.\n\n**Example:** *Explain how the internet works* in 40 words, simple style.\n\n**Steps**\n1. Press **📏 40 words + ELI5** → Send.\n2. Open the Evaluation panel: *Word limit* should show ✅.\n3. Press **🌐 Reply in Tamil** → Send. Expect a Tamil answer within 60 words.\n4. Change *Max words* to 15 and Send again to see a tighter answer."
                            )
                            max_words = gr.Number(
                                label="Max words",
                                value=0,
                                precision=0,
                                minimum=0,
                                info="0 = no limit"
                            )
                            answer_language = gr.Dropdown(
                                label="Answer language",
                                choices=LANGUAGES,
                                value="Same as user"
                            )
                            answer_style = gr.Dropdown(
                                label="Answer style",
                                choices=list(STYLES.keys()),
                                value="Default"
                            )
                            extra_rules = gr.Textbox(
                                label="Extra rules",
                                placeholder="Example: Use one real-life analogy. No jargon.",
                                lines=2
                            )
                            with gr.Row():
                                ex_limit = example_button("📏 40 words + ELI5")
                                ex_tamil = example_button("🌐 Reply in Tamil")

                        # ---- 9-11. Structured / JSON / Table ----
                        with gr.Accordion(
                            "🧱 9-11 · Structured · JSON · Table Output",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** get the answer in an exact format (JSON, table, steps).\n\n**Example:** Machine Learning details as JSON; languages compared as a table.\n\n**Steps**\n1. Press **{ } JSON** → Send. Expect a code block with the 4 fields; Evaluation shows *JSON valid ✅*.\n2. Press **▦ Table** → Send. Expect a real table with 4 columns; Evaluation shows *Markdown table ✅*.\n3. Press **1. Steps** → Send. Expect a numbered deployment guide.\n4. Edit *Fields / columns* and Send to change the structure.'
                            )
                            output_format = gr.Radio(
                                label="Output format",
                                choices=OUTPUT_FORMATS,
                                value="Normal Text"
                            )
                            format_hint = gr.Textbox(
                                label="Fields / columns (optional)",
                                placeholder="name, definition, use_case, difficulty",
                                lines=1
                            )
                            with gr.Row():
                                ex_json = example_button("{ } JSON")
                                ex_table = example_button("▦ Table")
                                ex_steps = example_button("1. Steps")

                        # ---- 12. Prompt comparison ----
                        with gr.Accordion(
                            "⚖️ 12 · Prompt Comparison",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** runs the *same question* with each chosen technique so you can compare the answers.\n\n**Example:** *Explain what a database index is.* with 7 techniques.\n\n**Steps**\n1. Press **📥 Load example: all techniques**.\n2. (Optional) Context tab → Max Output Tokens 800+.\n3. Press **⚖️ Compare Techniques**.\n4. Wait: the chat adds one section per technique.\n5. Read the summary table on the right (time, tokens, words, checks).\n6. Note which technique was fastest, shortest and clearest.'
                            )
                            compare_techniques = gr.CheckboxGroup(
                                label="Techniques to compare",
                                choices=COMPARE_CHOICES,
                                value=COMPARE_DEFAULT
                            )
                            compare_btn = gr.Button(
                                "⚖️ Compare Techniques",
                                variant="primary",
                                elem_classes=["compare-btn"]
                            )
                            ex_compare = example_button("📥 Load example: all techniques")

                        # ---- 13. Response evaluation ----
                        with gr.Accordion(
                            "📊 13 · Response Evaluation",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                '**What:** the right panel reports speed, tokens, the prompt settings used and automatic checks. You rate quality 1-5.\n\n**Example:** the same sentiment question with Zero-Shot vs Few-Shot.\n\n**Steps**\n1. Press **📥 Load evaluation test** → Send (Zero-Shot).\n2. Close the tab and read the report. Note the answer.\n3. Open Prompt → Press **🔢 Few-Shot** → Send the same text.\n4. Compare the two reports: *Technique*, *Examples used*, tokens and answers.\n5. Rate Relevance, Accuracy, Completeness, Consistency from 1 to 5.'
                            )
                            ex_eval = example_button("📥 Load evaluation test")

    # ---------- Header tabs (replace the settings icon) ----------
    tab_outputs = [
        settings_column,
        evaluation_display,
        active_tab,
        context_tab,
        prompt_tab,
        context_tab_btn,
        prompt_tab_btn,
        eval_tab_btn
    ]

    eval_tab_btn.click(fn=click_eval_tab, inputs=[active_tab], outputs=tab_outputs)
    context_tab_btn.click(fn=click_context_tab, inputs=[active_tab], outputs=tab_outputs)
    prompt_tab_btn.click(fn=click_prompt_tab, inputs=[active_tab], outputs=tab_outputs)

    # ---------- Example buttons ----------
    example_outputs = [
        system_prompt, role_choice, technique, example_shots, task_instruction,
        max_words, answer_language, answer_style, extra_rules,
        output_format, format_hint, compare_techniques, message
    ]

    example_buttons = [
        (ex_system, "system"),
        (ex_teacher, "teacher"),
        (ex_developer, "developer"),
        (ex_analyst, "analyst"),
        (ex_zero, "zero"),
        (ex_one, "one"),
        (ex_few, "few"),
        (ex_instruction, "instruction"),
        (ex_limit, "limit"),
        (ex_tamil, "tamil"),
        (ex_json, "json"),
        (ex_table, "table"),
        (ex_steps, "steps"),
        (ex_compare, "compare"),
        (ex_eval, "evaluation"),
    ]

    for button, key in example_buttons:
        button.click(fn=EXAMPLES[key], inputs=[], outputs=example_outputs)

    reset_btn.click(
        fn=reset_prompt_settings,
        inputs=[],
        outputs=[
            system_prompt, role_choice, technique, example_shots, task_instruction,
            max_words, answer_language, answer_style, extra_rules,
            output_format, format_hint, compare_techniques
        ]
    )

    sample_prompt.change(fn=pick_sample_prompt, inputs=[sample_prompt], outputs=[message])

    # ---------- Chat inputs / outputs ----------
    chat_inputs = [
        message,
        chat_history,
        user_name,
        user_role,
        skill_level,
        interests,
        use_user_context,
        reference_context,
        temperature,
        max_output_tokens,
        top_p,
        system_prompt,
        role_choice,
        technique,
        example_shots,
        task_instruction,
        max_words,
        answer_language,
        answer_style,
        extra_rules,
        output_format,
        format_hint
    ]

    chat_outputs = [
        chat_display,
        chat_history,
        message,
        evaluation_display
    ]

    submit_btn.click(fn=chat_with_gemini, inputs=chat_inputs, outputs=chat_outputs)
    message.submit(fn=chat_with_gemini, inputs=chat_inputs, outputs=chat_outputs)

    # ---------- Prompt comparison ----------
    compare_inputs = [
        message,
        chat_history,
        active_tab,
        user_name,
        user_role,
        skill_level,
        interests,
        use_user_context,
        reference_context,
        temperature,
        max_output_tokens,
        top_p,
        system_prompt,
        role_choice,
        example_shots,
        task_instruction,
        max_words,
        answer_language,
        answer_style,
        extra_rules,
        output_format,
        format_hint,
        compare_techniques
    ]

    compare_outputs = [
        chat_display,
        chat_history,
        message,
        evaluation_display,
        settings_column,
        active_tab,
        context_tab_btn,
        prompt_tab_btn,
        eval_tab_btn
    ]

    compare_btn.click(fn=compare_prompts, inputs=compare_inputs, outputs=compare_outputs)

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chat_display,
            chat_history,
            message,
            evaluation_display
        ]
    )

    demo.load(fn=None, inputs=None, outputs=None, js=autoscroll_js)


# ============================================
# 12. START APPLICATION
# ============================================

if __name__ == "__main__":
    # Local: http://127.0.0.1:7860
    # Render: Render supplies PORT and needs 0.0.0.0
    is_render = bool(os.environ.get("RENDER")) or bool(os.environ.get("PORT"))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = int(os.environ.get("PORT", 7860))

    print(f"Starting NeuroGen AI on {host}:{port}")

    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=False
    )