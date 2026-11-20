# ============================================
# NEUROGEN AI ASSISTANT  (corrected version)
# Python + Google Gemini API + Gradio
# Compact one-screen layout | Local + Render
# Header tabs: Evaluation | Context | Prompt | Memory (Memory + RAG Lab)
# ============================================

import os
import re
import json
import pandas as pd
import threading
import numpy as np
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

# ---------- Model selection (default + dynamic dropdown) ----------
DEFAULT_MODEL = MODEL

# Built-in list, used instantly at start-up and when the live API list fails.
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

_SKIP_WORDS = ("embedding", "tts", "image", "live", "aqa", "audio", "vision", "robotics", "computer-use")


def unique_models(names):
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_model_choices():
    """Ask the Gemini API which models can generate text. Default model stays first."""
    live = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if not name.startswith("gemini"):
                continue
            if actions and "generateContent" not in actions:
                continue
            if any(w in name for w in _SKIP_WORDS):
                continue
            live.append(name)
    except Exception as e:
        print("Model list error:", str(e))

    if live:
        return unique_models([DEFAULT_MODEL] + sorted(live)), True
    return unique_models(FALLBACK_MODELS), False


def model_status_text(count, from_api):
    source = "loaded live from Gemini API" if from_api else "built-in list (API list unavailable)"
    return f"Default: `{DEFAULT_MODEL}` · {count} models · {source}"


def refresh_models(current):
    names, from_api = get_model_choices()
    return (
        gr.update(choices=names, value=(current or DEFAULT_MODEL)),
        model_status_text(len(names), from_api),
    )


def use_default_model():
    return DEFAULT_MODEL


# ---------- Evaluation graph data (free: Gradio BarPlot + pandas) ----------
METRICS = ["Latency (s)", "Output tokens", "Words"]


def add_run(log, name, model, latency, out_tokens, text):
    log = list(log or [])
    no = (log[-1]["No"] + 1) if log else 1
    log.append({
        "No": no,
        "Run": f"#{no} {name}",
        "Model": model,
        "Latency (s)": round(float(latency), 2),
        "Output tokens": int(out_tokens or 0),
        "Words": len((text or "").split()),
    })
    return log[-20:]


def make_plot_df(log, metric):
    # FIX 2: never return an empty DataFrame (BarPlot can crash on it)
    metric = metric if metric in METRICS else METRICS[0]
    rows = [{"Run": r["Run"], "Value": float(r.get(metric, 0))} for r in (log or [])]
    if not rows:
        rows = [{"Run": "No runs yet", "Value": 0.0}]
    return pd.DataFrame(rows, columns=["Run", "Value"])


def update_plot(log, metric):
    return make_plot_df(log, metric)


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
# 2B. MEMORY + DOCUMENT Q&A (RAG)
# ============================================

# ---------- Memory: facts kept in a JSON file, so they survive
# "Clear Conversation" and app restarts ----------

MEMORY_FILE = os.getenv("MEMORY_FILE", "neurogen_memory.json")
MEMORY_LIMIT = 50
_memory_lock = threading.Lock()

SAMPLE_MEMORY = (
    "My name is Arun. I am preparing for a data science interview. "
    "I prefer short answers with Python examples."
)
SAMPLE_MEMORY_QUESTION = "What is my name, and what am I preparing for?"

_REMEMBER_RE = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that|\s*:)?\s+(.+)$", re.I | re.S)


def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [m for m in data if isinstance(m, dict) and m.get("text")]
    except Exception:
        return []


def save_memory(items):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-MEMORY_LIMIT:], f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print("Memory save error:", str(e))
        return False


def add_memory_item(text):
    text = " ".join((text or "").split())[:400]
    if not text:
        return False, "Nothing to save."
    with _memory_lock:
        items = load_memory()
        if any(m["text"].lower() == text.lower() for m in items):
            return False, "Already saved."
        items.append({"text": text, "saved": datetime.now().strftime("%d-%m-%Y %H:%M")})
        ok = save_memory(items)
    return ok, ("Saved." if ok else "Could not write the memory file.")


def delete_memory_item(label):
    match = re.match(r"\s*(\d+)\.", label or "")
    if not match:
        return False
    index = int(match.group(1)) - 1
    with _memory_lock:
        items = load_memory()
        if 0 <= index < len(items):
            items.pop(index)
            return save_memory(items)
    return False


def extract_memory_command(message):
    """'Remember that my name is Arun' -> 'my name is Arun'"""
    m = _REMEMBER_RE.match(message or "")
    return m.group(1).strip() if m else None


def memory_prompt_text(items):
    return "\n".join(f"- {m['text']}" for m in (items or [])[-30:])


def memory_ui_update():
    items = load_memory()
    if items:
        lines = "\n".join(f"{i}. {m['text']}" for i, m in enumerate(items, start=1))
        md = f"**{len(items)} saved facts** · file: `{MEMORY_FILE}`\n\n{lines}"
    else:
        md = f"No saved memory yet. · file: `{MEMORY_FILE}`"
    labels = [f"{i}. {m['text'][:70]}" for i, m in enumerate(items, start=1)]
    return md, gr.update(choices=labels, value=None)


def ui_add_memory(text):
    ok, msg = add_memory_item(text)
    if not ok:
        gr.Warning(msg)
    md, dd = memory_ui_update()
    return ("" if ok else text), md, dd


def ui_add_sample_memory():
    add_memory_item(SAMPLE_MEMORY)
    md, dd = memory_ui_update()
    return "", md, dd


def ui_delete_memory(label):
    if not delete_memory_item(label):
        gr.Warning("Pick a memory from the list first.")
    md, dd = memory_ui_update()
    return gr.update(), md, dd


def ui_clear_memory():
    with _memory_lock:
        save_memory([])
    md, dd = memory_ui_update()
    return gr.update(), md, dd


def ui_save_last_answer(history):
    """Retain SELECTED information: store the last AI answer as a memory."""
    last = ""
    for item in reversed(history or []):
        if item.get("role") == "assistant":
            last = str(item.get("content", ""))
            break
    last = re.sub(r"```[a-z]*", "", last).strip()
    if not last:
        gr.Warning("There is no AI answer to save yet.")
    else:
        add_memory_item("Earlier answer: " + last[:300])
    md, dd = memory_ui_update()
    return gr.update(), md, dd


# ---------- RAG: upload -> chunk -> embed -> retrieve -> answer ----------

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
MAX_CHUNKS = 400
TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".log")

SAMPLE_DOC_NAME = "novatech_handbook.txt"
SAMPLE_DOC_TEXT = """NovaTech Employee Handbook (sample document)

Annual Leave: Every full-time employee gets 24 days of paid annual leave per year. Up to 8 unused days can be carried over to the next year. Leave requests must be approved by the reporting manager at least 5 working days in advance.

Sick Leave: Employees receive 12 days of paid sick leave each year. A medical certificate is required if the sick leave is longer than 3 consecutive days.

Remote Work: Employees may work from home up to 3 days per week. Tuesday and Thursday are mandatory office days for team meetings and planning sessions.

Equipment: Each employee receives a company laptop on the joining day. Laptops are replaced every 36 months. Lost, stolen or damaged equipment must be reported to the IT helpdesk within 24 hours.

Working Hours: Core working hours are 10:00 AM to 4:00 PM. The standard working time is 8 hours per day, Monday to Friday. Overtime must be pre-approved and is compensated with time off.

Learning Budget: Every employee has an annual learning budget of 30,000 rupees for courses, certifications and books. Requests must be approved by the reporting manager before purchase.

Expense Claims: Travel and meal expenses must be submitted within 15 days with receipts. Claims above 10,000 rupees need additional approval from the finance team.

Health Insurance: The company provides health insurance for the employee, spouse and up to two children. Coverage starts from the first day of employment and includes 5 lakh rupees of annual cover.
"""

SAMPLE_RAG_Q1 = "How many days of annual leave do employees get, and how many can be carried over?"
SAMPLE_RAG_Q2 = "How often is a laptop replaced, and what should I do if it is lost?"
SAMPLE_RAG_Q3 = "What is the salary of the CEO?"   # NOT in the document

_STOPWORDS = set(
    "a an the is are was were be been of to in on at for and or with by from as it this that "
    "do does did how what when where which who whom why can could should would will i you we they "
    "my our your their me us if than then so not no up per each every".split()
)


def empty_index():
    return {"chunks": [], "vectors": None, "mode": "none", "docs": []}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOPWORDS]


def md_cell(text):
    return str(text).replace("|", "/").replace("\n", " ").strip()


def read_document(path):
    """Return (text, error). TXT/MD/CSV always; PDF needs pypdf; DOCX needs python-docx."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in TEXT_EXTENSIONS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), None
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                return "", "PDF needs the free package pypdf (pip install pypdf)"
            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages), None
        if ext == ".docx":
            try:
                import docx
            except ImportError:
                return "", "DOCX needs the free package python-docx (pip install python-docx)"
            return "\n".join(p.text for p in docx.Document(path).paragraphs), None
        return "", f"unsupported file type {ext or '(none)'}"
    except Exception as e:
        return "", f"could not read file ({e})"


def chunk_text(text, size, overlap):
    words = (text or "").split()
    size = max(10, int(size or 80))
    overlap = min(max(0, int(overlap or 0)), size - 1)
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def embed_texts(texts, task_type):
    """Free Gemini embeddings. Returns (normalised float32 matrix, error)."""
    try:
        vectors = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=genai.types.EmbedContentConfig(task_type=task_type)
            )
            vectors.extend([list(e.values) for e in result.embeddings])
        if len(vectors) != len(texts):
            return None, "embedding count mismatch"
        arr = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms, None
    except Exception as e:
        return None, str(e)


def keyword_scores(chunks, query):
    """BM25-style keyword scoring (used when embeddings are unavailable)."""
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avg_len = (sum(len(d) for d in docs) / n) if n else 1.0
    avg_len = avg_len or 1.0
    q_terms = set(tokenize(query))
    k1, b = 1.5, 0.75
    scores = np.zeros(n, dtype="float32")
    for term in q_terms:
        df = sum(1 for d in docs if term in d)
        if df == 0:
            continue
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(term)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    top = float(scores.max()) if n else 0.0
    return scores / top if top > 0 else scores


def add_documents(index, items, chunk_size, overlap):
    """items = [(name, text)]. Returns (new_index, notes)."""
    index = index or empty_index()
    new = {
        "chunks": list(index["chunks"]),
        "vectors": index["vectors"],
        "mode": index["mode"],
        "docs": list(index["docs"]),
    }
    notes = []
    fresh = []

    for name, text in items:
        if any(d["name"] == name for d in new["docs"]):
            notes.append(f"⏭ {name}: already indexed")
            continue
        pieces = chunk_text(text, chunk_size, overlap)
        if not pieces:
            notes.append(f"⚠️ {name}: no readable text")
            continue
        room = MAX_CHUNKS - len(new["chunks"]) - len(fresh)
        if room <= 0:
            notes.append(f"⚠️ {name}: skipped (limit of {MAX_CHUNKS} chunks reached)")
            continue
        pieces = pieces[:room]
        for n, piece in enumerate(pieces, start=1):
            fresh.append({"source": name, "n": n, "text": piece})
        new["docs"].append({"name": name, "chunks": len(pieces)})
        notes.append(f"✅ {name}: {len(pieces)} chunks")

    if not fresh:
        return new, notes

    # FIX 5: only call the embedding API when the vectors can actually be used
    can_embed = new["vectors"] is not None or not new["chunks"]
    if can_embed:
        vectors, err = embed_texts([c["text"] for c in fresh], "RETRIEVAL_DOCUMENT")
    else:
        vectors, err = None, None

    if vectors is not None and can_embed:
        new["vectors"] = vectors if new["vectors"] is None else np.vstack([new["vectors"], vectors])
        new["mode"] = "embeddings"
    else:
        new["vectors"] = None
        new["mode"] = "keyword"
        if err:
            notes.append("⚠️ Embeddings unavailable, using keyword search. " + md_cell(err)[:120])
    new["chunks"].extend(fresh)
    return new, notes


def index_status_md(index, notes=None):
    index = index or empty_index()
    if not index["chunks"]:
        text = "No documents indexed yet."
    else:
        mode = (
            f"Gemini embeddings (`{EMBED_MODEL}`)" if index["mode"] == "embeddings"
            else "keyword (BM25)"
        )
        docs = ", ".join(f"{d['name']} ({d['chunks']})" for d in index["docs"])
        text = (
            f"**{len(index['docs'])} document(s) · {len(index['chunks'])} chunks** · "
            f"search: {mode}\n\n{docs}"
        )
    if notes:
        text += "\n\n" + "\n".join(f"- {n}" for n in notes)
    return text


def retrieve(index, query, top_k):
    chunks = (index or {}).get("chunks") or []
    if not chunks:
        return [], "none"
    scores, mode = None, "keyword"
    vectors = index.get("vectors")
    if vectors is not None and len(vectors) == len(chunks):
        qv, _err = embed_texts([query], "RETRIEVAL_QUERY")
        if qv is not None:
            scores = vectors @ qv[0]
            mode = "embeddings"
    if scores is None:
        scores = keyword_scores(chunks, query)
        mode = "keyword"
    order = np.argsort(-scores)[:max(1, int(top_k or 3))]
    hits = [
        {
            "rank": r,
            "source": chunks[i]["source"],
            "n": chunks[i]["n"],
            "score": float(scores[i]),
            "text": chunks[i]["text"],
        }
        for r, i in enumerate(order, start=1)
    ]
    return hits, mode


def format_rag_context(hits):
    return "\n\n".join(
        f"[{h['rank']}] (source: {h['source']}, chunk {h['n']})\n{h['text']}" for h in hits
    )


def retrieval_md(hits, mode):
    if not hits:
        return ""
    label = "Gemini embeddings (cosine similarity)" if mode == "embeddings" else "keyword BM25 (0-1)"
    rows = "\n".join(
        f"| {h['rank']} | {md_cell(h['source'])} #{h['n']} | {h['score']:.3f} | {md_cell(h['text'][:110])}... |"
        for h in hits
    )
    return f"""
### 📚 Retrieved Chunks · {label}

| Rank | Source | Score | Preview |
|---|---|---|---|
{rows}

**Retrieval quality (manual):** Is the correct chunk ranked #1? Do the top chunks really contain the answer? Is the answer faithful to them?
"""


def make_filler(*texts):
    def filler():
        return texts if len(texts) > 1 else texts[0]
    return filler


def _file_paths(files):
    paths = []
    for f in files or []:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if path:
            paths.append(path)
    return paths


def ui_index_files(files, chunk_size, overlap, index):
    items, notes = [], []
    paths = _file_paths(files)
    if not paths:
        gr.Warning("Upload at least one file first.")
        return index, index_status_md(index)
    for path in paths:
        text, err = read_document(path)
        name = os.path.basename(path)
        if err:
            notes.append(f"⚠️ {name}: {err}")
        else:
            items.append((name, text))
    new_index, add_notes = add_documents(index, items, chunk_size, overlap)
    return new_index, index_status_md(new_index, notes + add_notes)


def ui_add_sample_doc(chunk_size, overlap, index):
    new_index, notes = add_documents(index, [(SAMPLE_DOC_NAME, SAMPLE_DOC_TEXT)], chunk_size, overlap)
    return new_index, index_status_md(new_index, notes)


def ui_clear_index():
    idx = empty_index()
    return idx, index_status_md(idx)


def ui_preview_retrieval(index, query, top_k):
    """Show the retrieved chunks WITHOUT calling the chat model (cheap retrieval test)."""
    if not (index and index.get("chunks")):
        return "Index a document first."
    if not (query or "").strip():
        return "Type a question to test retrieval."
    t0 = time.perf_counter()
    hits, mode = retrieve(index, query.strip(), int(top_k or 3))
    took = time.perf_counter() - t0
    return retrieval_md(hits, mode) + f"\n*Retrieval took {took:.2f}s*"


def memory_hit(on_answer, off_answer, facts, question):
    """Heuristic: does the memory-ON answer use words from saved facts that the OFF answer lacks?"""
    q = set(tokenize(question))
    fact_words = {w for f in facts for w in tokenize(f) if len(w) >= 4} - q
    return bool((set(tokenize(on_answer)) - set(tokenize(off_answer))) & fact_words)


def create_context_compare_report(kind, question, results, model_name, extra_md=""):
    title = "🧠 Memory ON vs OFF" if kind == "memory" else "📚 Documents (RAG) ON vs OFF"
    short_q = md_cell(question)[:110]
    rows = []
    for r in results:
        if r["status"] == "ok":
            tokens = r["out_tokens"] if r["out_tokens"] is not None else "-"
            rows.append(f"| {r['name']} | {r['latency']:.2f} | {tokens} | {r['words']} | {r['check']} |")
        else:
            rows.append(f"| {r['name']} | - | - | - | ❌ error |")
    table = "\n".join(rows)

    if kind == "memory":
        observe = (
            "| Look for | OFF | ON |\n|---|---|---|\n"
            "| Knows your saved facts | No (guesses or asks) | Yes |\n"
            "| Personalised (name, goal, style) | No | Yes |\n"
            "| Invents facts | Possible | Uses only saved facts |"
        )
    else:
        observe = (
            "| Look for | OFF | ON |\n|---|---|---|\n"
            "| Answer from your document | No (general guess) | Yes, with [n] citations |\n"
            "| Company-specific numbers | Often wrong | Match the document |\n"
            "| Question not in document | Guesses | Says it is not in the documents |"
        )

    return f"""
## ⚖️ {title}

**Question:** {short_q}

**Model:** {model_name}

| Mode | Time (s) | Out tokens | Words | Check |
|---|---|---|---|---|
{table}

### 🧠 What to observe

{observe}

Rate each answer **1-5** for Relevance, Accuracy and Faithfulness. Read both full answers in the chat.
{extra_md}
*Quality ratings require human review.*
"""


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
    use_user_context, reference_context, technique_block,
    memory_text="", rag_text=""
):
    # FIX 3: the profile text must never contradict the Saved Memory block
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
        if memory_text:
            user_context += (
                "\nIf a profile field above says 'Not provided', check the Saved Memory "
                "section before saying the information is unknown.\n"
            )
    elif memory_text:
        user_context = """
The User Profile form is disabled.
Do not assume profile details, but DO use the Saved Memory facts below.
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

    if rag_text:
        extra_reference = ""
        if reference_context and reference_context.strip():
            extra_reference = f"\nAdditional Reference Information:\n{reference_context.strip()}\n"
        grounding_context = f"""
Retrieved Document Excerpts:
{rag_text}
{extra_reference}
Instructions:
1. Answer using ONLY the document excerpts above.
2. Cite the excerpt number like [1] after each fact you use.
3. If the excerpts do not contain the answer, clearly say:
"The uploaded documents do not contain this information."
"""

    memory_section = ""
    if memory_text:
        memory_section = f"""
Saved Memory (facts the user asked you to remember between conversations):
{memory_text}
Use these facts when they are relevant to the question.
"""

    technique_section = ""
    if technique_block:
        technique_section = f"\nPrompting Technique Context:\n{technique_block}\n"

    return f"""
{user_context}
Dynamic Context:
{dynamic_context}
{technique_section}{memory_section}
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
    settings_rows=None, check_rows=None, extra_md=""
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
{settings_md}{checks_md}{extra_md}
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


def create_comparison_report(question, results, model_name=None):
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

**Model:** {model_name or DEFAULT_MODEL}

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
    format_hint,
    model_name,
    run_log,
    use_memory,
    use_rag,
    top_k,
    rag_index
):
    history = history or []
    used_model = model_name or DEFAULT_MODEL

    if not message or not message.strip():
        yield show_chat(history), history, "", "Please enter a question.", run_log
        return

    message = message.strip()
    reference_provided = bool(reference_context and reference_context.strip())

    technique_block = build_technique_block(technique, example_shots, task_instruction)

    # ---- Memory: auto-save "remember ..." messages, then load saved facts ----
    memory_items = []
    saved_fact = None
    if use_memory:
        fact = extract_memory_command(message)
        if fact:
            saved, _msg = add_memory_item(fact)
            saved_fact = fact if saved else None
        memory_items = load_memory()
    memory_text = memory_prompt_text(memory_items)

    # ---- RAG: retrieve the best chunks from the uploaded documents ----
    hits, rag_mode, rag_text, rag_seconds = [], "", "", 0.0
    if use_rag and rag_index and rag_index.get("chunks"):
        t_rag = time.perf_counter()
        hits, rag_mode = retrieve(rag_index, message, int(top_k or 3))
        rag_seconds = time.perf_counter() - t_rag
        rag_text = format_rag_context(hits)
        rag_label = f"ON · {rag_mode} · top-{int(top_k or 3)} · {rag_seconds:.2f}s retrieval"
    elif use_rag:
        rag_label = "ON · ⚠️ no documents indexed"
    else:
        rag_label = "OFF"

    contextual_message = build_contextual_message(
        message, user_name, user_role, skill_level, interests,
        use_user_context, reference_context, technique_block,
        memory_text, rag_text
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
        "⏳ Generating response and measuring performance...",
        run_log
    )

    try:
        answer = ""
        prompt_tokens = None
        response_tokens = None
        start_time = time.perf_counter()

        stream = client.models.generate_content_stream(
            model=used_model,
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
                    "⏳ Response is being generated...",
                    run_log
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
            ("Model", used_model + (" (default)" if used_model == DEFAULT_MODEL else "")),
            ("Memory", f"ON · {len(memory_items)} facts" if use_memory else "OFF"),
            ("Documents (RAG)", rag_label),
            ("Technique", technique),
            ("Examples used", describe_examples(technique, example_shots)),
            ("Role", role_choice),
            ("Output format", output_format),
            ("Word limit", str(word_limit) if word_limit > 0 else "None"),
            ("Style", answer_style),
            ("Language", answer_language),
            ("System prompt", "Custom" if (system_prompt or "").strip() else "Default"),
        ]

        if saved_fact:
            settings_rows.append(("Auto-saved to memory", md_cell(saved_fact)[:80]))
        if rag_text:
            cited = bool(re.search(r"\[\d+\]", answer))
            check_rows.append(("Cites a source [n]", "✅ Yes" if cited else "❌ No"))
        extra_md = retrieval_md(hits, rag_mode) if hits else ""

        evaluation_report = create_evaluation_report(
            latency, prompt_tokens, response_tokens, reference_provided,
            settings_rows, check_rows, extra_md
        )

        new_log = add_run(run_log, used_model, used_model, latency, response_tokens, answer)

        yield show_chat(final_history), final_history, "", evaluation_report, new_log

    except Exception as e:
        print("Gemini Error:", str(e))
        error_history = base_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]
        yield (
            show_chat(error_history),
            error_history,
            "",
            f"Evaluation unavailable because an error occurred.\n\nError: {html.escape(str(e))}",
            run_log
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
    techniques,
    model_name,
    run_log
):
    history = history or []
    used_model = model_name or DEFAULT_MODEL

    def progress(h):
        return (
            show_chat(h), h, "",
            gr.update(), gr.update(), active_value, gr.update(), gr.update(), gr.update(), gr.update(),
            run_log, gr.update()
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
                    model=used_model,
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
                run_log = add_run(run_log, technique, used_model, latency, out_tokens, text)

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
    report = create_comparison_report(message, results, used_model)

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
        gr.update(variant="secondary"),
        run_log,
        gr.update(visible=True),
    )


# ============================================
# 8B. COMPARE WITH vs WITHOUT (memory / documents)
# ============================================

def compare_context_modes(
    kind, message, history, active_value, temperature, max_output_tokens,
    top_p, model_name, run_log, rag_index, top_k
):
    """Same question twice: feature OFF, then feature ON. User profile is switched off
    so the only difference between the two answers is memory / retrieved documents."""
    history = history or []
    used_model = model_name or DEFAULT_MODEL
    label = "Memory" if kind == "memory" else "Documents"

    def progress(h):
        return (
            show_chat(h), h, "",
            gr.update(), gr.update(), active_value, gr.update(), gr.update(), gr.update(),
            gr.update(), run_log, gr.update()
        )

    if not message or not message.strip():
        gr.Warning("Type or load a question in the message box first.")
        yield progress(history)
        return

    message = message.strip()
    memory_items = load_memory()
    hits, rag_mode, rag_text = [], "", ""

    if kind == "memory":
        if not memory_items:
            gr.Warning("No saved memory yet. Add a fact in the Memory tab first.")
            yield progress(history)
            return
    else:
        if not (rag_index and rag_index.get("chunks")):
            gr.Warning("Index a document first (Memory tab → Documents + RAG).")
            yield progress(history)
            return
        hits, rag_mode = retrieve(rag_index, message, int(top_k or 3))
        rag_text = format_rag_context(hits)

    base_history = history + [{"role": "user", "content": message}]
    system_instruction = build_system_instruction(
        "", "None", 0, "Same as user", "Default", "", "Normal Text", ""
    )
    variants = [
        (f"{label} OFF", "", ""),
        (f"{label} ON", memory_prompt_text(memory_items) if kind == "memory" else "", rag_text),
    ]

    sections, results, answers = [], [], {}
    yield progress(base_history + [{"role": "assistant", "content": f"⏳ Running {label} ON vs OFF..."}])

    for name, mem_text, doc_text in variants:
        contextual = build_contextual_message(
            message, "", "", "Beginner", "", False, "", "", mem_text, doc_text
        )
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model=used_model,
                contents=[{"role": "user", "parts": [{"text": contextual}]}],
                config=make_config(system_instruction, temperature, max_output_tokens, top_p, "Normal Text")
            )
            latency = time.perf_counter() - start
            text = (getattr(response, "text", None) or "").strip() or "No response received."
            usage = getattr(response, "usage_metadata", None)
            out_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            words = len(text.split())
            answers[name] = text

            check = "-"
            if name.endswith("ON"):
                if kind == "memory":
                    hit = memory_hit(text, answers.get(f"{label} OFF", ""),
                                     [m["text"] for m in memory_items], message)
                    check = "✅ uses saved facts" if hit else "❌ no saved fact seen"
                else:
                    cited = bool(re.search(r"\[\d+\]", text))
                    check = "✅ cites [n]" if cited else "❌ no citation"

            results.append({"name": name, "status": "ok", "latency": latency,
                            "out_tokens": out_tokens, "words": words, "check": check})
            run_log = add_run(run_log, name, used_model, latency, out_tokens, text)
            sections.append(f"### 🔹 {name}\n{text}\n\n*{latency:.2f}s · {words} words*")
        except Exception as e:
            print("Gemini Compare Error:", str(e))
            results.append({"name": name, "status": "error"})
            sections.append(f"### 🔹 {name}\n❌ Error: {e}")

        yield progress(base_history + [{"role": "assistant", "content": "\n\n".join(sections)}])

    final_history = base_history + [{"role": "assistant", "content": "\n\n".join(sections)}]
    report = create_context_compare_report(
        kind, message, results, used_model, retrieval_md(hits, rag_mode) if hits else ""
    )

    yield (
        show_chat(final_history), final_history, "",
        gr.update(value=report, visible=True),
        gr.update(visible=False),
        "evaluation",
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
        gr.update(variant="primary"),
        gr.update(variant="secondary"),
        run_log,
        gr.update(visible=True),
    )


def compare_memory(*args):
    yield from compare_context_modes("memory", *args)


def compare_rag(*args):
    yield from compare_context_modes("rag", *args)


# ============================================
# 9. CLEAR CONVERSATION, TABS, EXAMPLE LOADERS
# ============================================

def clear_chat():
    return (
        show_chat([]),
        [],
        "",
        "## 📊 Chatbot Evaluation\n\nEvaluation results will appear here.",
        []
    )


def switch_tab(target, current):
    """
    Header tabs: Evaluation | Context | Prompt | Memory.
    - Evaluation tab always goes back to the evaluation report.
    - Clicking Context / Prompt / Memory opens that panel on the right.
    - Clicking the already-active tab also returns to Evaluation.
    """
    new = "evaluation" if (target == "evaluation" or current == target) else target
    show_panel = new in ("context", "prompt", "memory")
    return (
        gr.update(visible=show_panel),                                         # settings_column
        gr.update(visible=not show_panel),                                     # eval_column
        new,                                                                   # active_tab state
        gr.update(visible=new == "context"),                                   # context_tab
        gr.update(visible=new == "prompt"),                                    # prompt_tab
        gr.update(visible=new == "memory"),                                    # memory_tab
        gr.update(variant="primary" if new == "context" else "secondary"),     # context_tab_btn
        gr.update(variant="primary" if new == "prompt" else "secondary"),      # prompt_tab_btn
        gr.update(variant="primary" if new == "memory" else "secondary"),      # memory_tab_btn
        gr.update(variant="primary" if new == "evaluation" else "secondary"),  # eval_tab_btn
    )


def click_context_tab(current):
    return switch_tab("context", current)


def click_prompt_tab(current):
    return switch_tab("prompt", current)


def click_memory_tab(current):
    return switch_tab("memory", current)


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
   SETTINGS PANEL — tabs + scrollable tab body
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

/* Accordions inside settings tabs */
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

/* Mobile: chat on top, right column below */
@media (max-width: 760px) {
    .gradio-container { padding: 5px !important; }
    #main-card { height: calc(100dvh - 10px) !important; max-height: calc(100dvh - 10px) !important; padding: 7px !important; gap: 5px !important; }
    #app-heading h1 { font-size: 16px !important; }
    #app-heading h3 { font-size: 9px !important; }
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

.pe-help, .pe-help p, .pe-help li {
    color: #AAB9D8 !important;
    font-size: 10px !important;
    line-height: 1.45 !important;
    margin: 0 0 3px 0 !important;
}
.pe-help b, .pe-help strong { color: #C5EFFF !important; }
.pe-help ol, .pe-help ul { padding-left: 16px !important; margin: 3px 0 !important; }

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

.quick-btn { white-space: normal !important; line-height: 1.2 !important; }

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


/* =====================================================
   EVALUATION TAB: model dropdown + graph + report
   ===================================================== */
#eval-column {
    flex: 1 1 0 !important;
    width: 100% !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    gap: 6px !important;
    background: linear-gradient(155deg, rgba(17,28,54,.98), rgba(9,15,31,.99)) !important;
    border: 1px solid rgba(137,105,255,.43) !important;
    border-radius: 11px !important;
    padding: 9px !important;
}
#eval-column::-webkit-scrollbar { width: 5px; }
#eval-column::-webkit-scrollbar-thumb { background: linear-gradient(#35D8FF,#855CFF); border-radius: 10px; }
#eval-column #evaluation-panel {
    flex: 0 0 auto !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    border: none !important;
    background: transparent !important;
    padding: 2px !important;
}
#eval-column .block { background: transparent !important; border-color: transparent !important; }
#eval-column input:not([type="radio"]):not([type="checkbox"]) {
    background: #080D1B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid rgba(89,156,255,.48) !important;
    border-radius: 7px !important;
    font-size: 11px !important;
}
#eval-column label, #eval-column label span, #eval-column .block-label { color: #F4F7FF !important; font-size: 10px !important; }
#eval-column [role="listbox"], #eval-column [role="option"] { background: #101A32 !important; color: #FFFFFF !important; }
#eval-column input[type="radio"] { accent-color: #35D8FF !important; }
#eval-plot { background: #0D1427 !important; border: 1px solid rgba(101,139,220,.35) !important; border-radius: 9px !important; min-height: 200px; }
#eval-plot, #eval-plot * { color: #DCE8FF !important; }
@media (max-width: 760px) {
    #eval-column { height: 100% !important; min-height: 0 !important; }
}


/* =====================================================
   TEXT + DROPDOWN COLOUR FIX  (dark theme everywhere)
   ===================================================== */
.gradio-container, .gradio-container.dark, .dark .gradio-container {
    --body-text-color: #F4F7FF !important;
    --body-text-color-subdued: #AAB9D8 !important;
    --block-title-text-color: #F4F7FF !important;
    --block-label-text-color: #F4F7FF !important;
    --block-label-background-fill: #101A32 !important;
    --block-info-text-color: #AAB9D8 !important;
    --block-background-fill: transparent !important;
    --block-border-color: rgba(89,156,255,.35) !important;
    --input-background-fill: #080D1B !important;
    --input-background-fill-focus: #0B1226 !important;
    --input-background-fill-hover: #0B1226 !important;
    --input-border-color: rgba(89,156,255,.48) !important;
    --input-border-color-focus: #35D8FF !important;
    --input-placeholder-color: #9EAFCE !important;
    --checkbox-label-background-fill: #101A32 !important;
    --checkbox-label-background-fill-hover: #18284A !important;
    --checkbox-label-background-fill-selected: #1F5FA8 !important;
    --checkbox-label-text-color: #F4F7FF !important;
    --checkbox-label-text-color-selected: #FFFFFF !important;
    --checkbox-label-border-color: rgba(89,156,255,.45) !important;
    --checkbox-label-border-color-hover: #35D8FF !important;
    --checkbox-label-border-color-selected: #35D8FF !important;
    --checkbox-background-color: #080D1B !important;
    --checkbox-background-color-selected: #35D8FF !important;
    --table-even-background-fill: #0F1930 !important;
    --table-odd-background-fill: #101A32 !important;
    --table-row-focus: #18284A !important;
    --background-fill-primary: #0D1224 !important;
    --background-fill-secondary: #111A31 !important;
    --color-accent-soft: #1B2B50 !important;
    --button-secondary-background-fill: #192C50 !important;
    --button-secondary-background-fill-hover: #24396B !important;
    --button-secondary-text-color: #E6F3FF !important;
}

#eval-column .wrap, #eval-column .wrap-inner, #eval-column .secondary-wrap,
#settings-column .wrap-inner, #settings-column .secondary-wrap,
#model-dropdown, #model-dropdown .wrap, #model-dropdown .wrap-inner, #model-dropdown .secondary-wrap,
#model-dropdown input, #model-dropdown ul, #model-dropdown li, #model-dropdown span {
    background: #080D1B !important;
    color: #F4F7FF !important;
    -webkit-text-fill-color: #F4F7FF !important;
    border-color: rgba(89,156,255,.48) !important;
}
#model-dropdown svg, #model-dropdown svg * { fill: #F4F7FF !important; color: #F4F7FF !important; }
ul.options, ul[role="listbox"] {
    background: #101A32 !important;
    border: 1px solid rgba(89,156,255,.5) !important;
    max-height: 220px !important;
}
ul.options li, ul[role="listbox"] li, [role="option"] {
    background: #101A32 !important;
    color: #F4F7FF !important;
}
ul.options li:hover, ul[role="listbox"] li:hover, [role="option"]:hover,
ul.options li.selected, ul[role="listbox"] li.selected, [role="option"][aria-selected="true"],
#model-dropdown li:hover, #model-dropdown li.selected {
    background: #1F4C8F !important;
    color: #FFFFFF !important;
}
#settings-column .wrap-inner input, #settings-column .secondary-wrap input {
    color: #F4F7FF !important;
    -webkit-text-fill-color: #F4F7FF !important;
}

#eval-column .wrap > label, #settings-column .wrap > label {
    background: #101A32 !important;
    color: #F4F7FF !important;
    border: 1px solid rgba(89,156,255,.45) !important;
}
#eval-column .wrap > label span, #settings-column .wrap > label span { color: #F4F7FF !important; background: transparent !important; }
#eval-column .wrap > label.selected, #settings-column .wrap > label.selected {
    background: linear-gradient(120deg, #147BC1, #4B54D8) !important;
    border-color: #35D8FF !important;
}
#eval-column .wrap > label.selected span, #settings-column .wrap > label.selected span { color: #FFFFFF !important; }

#metric-radio, #metric-radio fieldset, #metric-radio .wrap {
    overflow: visible !important;
    max-height: none !important;
    height: auto !important;
    background: transparent !important;
}
#metric-radio legend, #metric-radio > span, #metric-radio span[data-testid="block-info"], #metric-radio .block-info {
    color: #C5EFFF !important;
    background: transparent !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

.pe-help code, #evaluation-panel code, #eval-column code, #settings-column code, .prose code {
    background: #0A1226 !important;
    color: #8FEAFF !important;
    border: 1px solid rgba(89,156,255,.35) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
}
.pe-help, .pe-help *, #retrieval-preview, #retrieval-preview * { color: #D5E2FA !important; }
.pe-help b, .pe-help strong, #retrieval-preview b, #retrieval-preview strong { color: #C5EFFF !important; }
.pe-help table, #retrieval-preview table { width: 100% !important; font-size: 9px !important; }
.pe-help th, #retrieval-preview th { background: #1B2B50 !important; color: #C9F5FF !important; }
.pe-help td, #retrieval-preview td { background: #101A32 !important; color: #F4F7FF !important; }

#model-row { flex-wrap: wrap !important; gap: 5px !important; align-items: center !important; }
#model-dropdown { flex: 1 1 100% !important; min-width: 100% !important; }
#model-row button { flex: 1 1 40% !important; min-width: 70px !important; }

#doc-files, #doc-files * { color: #F4F7FF !important; }
#doc-files .block, #doc-files [data-testid="file-upload"], #doc-files .file-preview, #doc-files table {
    background: #101A32 !important;
}
#doc-files button { background: #192C50 !important; border: 1px solid rgba(89,156,255,.45) !important; }

#tab-row { flex-wrap: wrap !important; justify-content: flex-end !important; }
.tab-btn { min-width: 70px !important; font-size: 10px !important; padding: 0 8px !important; }

#memory-tab {
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
#memory-tab::-webkit-scrollbar { width: 5px; }
#memory-tab::-webkit-scrollbar-thumb { background: linear-gradient(#35D8FF,#855CFF); border-radius: 10px; }

/* FIX 4: hidden panels must really hide (the display:flex !important rules
   above would otherwise keep them visible) */
#settings-column.hide, #eval-column.hide, #context-tab.hide,
#prompt-tab.hide, #memory-tab.hide,
#settings-column.hidden, #eval-column.hidden, #context-tab.hidden,
#prompt-tab.hidden, #memory-tab.hidden { display: none !important; }
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
    "system": example_loader(
        system_prompt=SYSTEM_EXAMPLE,
        message=SYSTEM_EXAMPLE_MESSAGE,
    ),
    "teacher": example_loader(role="Teacher", message="Explain what an API is."),
    "developer": example_loader(
        role="Developer",
        message="Write a Python function to remove duplicates from a list while keeping the order.",
    ),
    "analyst": example_loader(
        role="Data Analyst",
        message="Our monthly sales dropped 15% last month. How should I investigate it?",
    ),
    "zero": example_loader(technique="Zero-Shot", message=ZERO_SHOT_MESSAGE),
    "one": example_loader(
        technique="One-Shot", shots=ONE_SHOT_EXAMPLE, message=SHOT_MESSAGE
    ),
    "few": example_loader(
        technique="Few-Shot", shots=SENTIMENT_SHOTS, message=SHOT_MESSAGE
    ),
    "instruction": example_loader(
        technique="Instruction-Based",
        instruction=INSTRUCTION_EXAMPLE,
        message=INSTRUCTION_MESSAGE,
    ),
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
    "compare": example_loader(
        role="Teacher",
        shots=COMPARE_SHOTS,
        instruction=COMPARE_INSTRUCTION,
        max_words=40,
        style="Simple (ELI5)",
        compare=list(COMPARE_CHOICES),
        message=COMPARE_MESSAGE,
    ),
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

        # Compact header: title left, four tabs right
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
                memory_tab_btn = gr.Button(
                    "🧠 Memory",
                    variant="secondary",
                    elem_classes=["tab-btn"],
                    elem_id="tab-memory-btn"
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
                # ---------- EVALUATION TAB: model + graph + report ----------
                with gr.Column(visible=True, elem_id="eval-column") as eval_column:

                    gr.Markdown("🤖 **Model**", elem_classes=["pe-help"])
                    with gr.Row(elem_id="model-row"):
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            show_label=False,
                            choices=unique_models(FALLBACK_MODELS),
                            value=DEFAULT_MODEL,
                            allow_custom_value=True,
                            scale=6,
                            container=False,
                            elem_id="model-dropdown"
                        )
                        refresh_models_btn = gr.Button(
                            "🔄",
                            size="sm",
                            scale=1,
                            min_width=36,
                            elem_classes=["quick-btn"]
                        )
                        default_model_btn = gr.Button(
                            "↩ Default",
                            size="sm",
                            scale=2,
                            min_width=70,
                            elem_classes=["quick-btn"]
                        )
                    model_status = gr.Markdown(
                        model_status_text(len(unique_models(FALLBACK_MODELS)), False),
                        elem_classes=["pe-help"]
                    )

                    metric_radio = gr.Radio(
                        label="Graph metric",
                        choices=METRICS,
                        value=METRICS[0],
                        elem_id="metric-radio"
                    )
                    eval_plot = gr.BarPlot(
                        value=make_plot_df([], METRICS[0]),
                        x="Run",
                        y="Value",
                        title="Run comparison",
                        x_title="Run",
                        y_title="Value",
                        height=200,
                        tooltip=["Run", "Value"],
                        elem_id="eval-plot"
                    )

                    evaluation_display = gr.Markdown(
                        """
                        ## 📊 Chatbot Evaluation

                        Evaluation results will appear here.
                        """,
                        elem_id="evaluation-panel"
                    )

                eval_log = gr.State([])
                rag_index = gr.State(empty_index())

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
                            # Default raised to 1000: newer Gemini "thinking" models
                            # count thinking tokens against this limit.
                            max_output_tokens = gr.Slider(
                                minimum=50,
                                maximum=4096,
                                value=1000,
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

                    # =====================================================
                    # TAB 4 : MEMORY + DOCUMENTS (RAG)
                    # =====================================================
                    with gr.Column(visible=False, elem_id="memory-tab") as memory_tab:

                        help_text(
                            "**Memory & RAG Lab.** Memory = facts the chatbot keeps *between conversations*. "
                            "RAG = answers taken from *your own documents*. Each box has numbered steps and ready-made examples."
                        )

                        # ---- A. Memory ----
                        with gr.Accordion(
                            "🧠 A · Memory (kept between conversations)",
                            open=True,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                "**What:** saved facts are stored in a file and added to every question. "
                                "They survive **Clear Conversation** and app restarts.\n\n"
                                "**Example facts:** *My name is Arun. I am preparing for a data science interview.*\n\n"
                                "**Steps**\n"
                                "1. Press **1️⃣ Add sample facts**. They appear in the list.\n"
                                "2. Press **🗑 Clear Conversation** (left side). The chat is empty, memory stays.\n"
                                "3. Press **2️⃣ Load recall question**, then **➤ Send**.\n"
                                "4. The answer should use your name and goal.\n"
                                "5. Press **⚖️ Compare Memory ON vs OFF** to see the difference.\n\n"
                                "**Tip:** you can also type *Remember that I live in Chennai* in the chat. It is saved automatically."
                            )
                            use_memory = gr.Checkbox(
                                label="Enable memory",
                                value=True,
                                info="Uses saved facts in answers and auto-saves 'remember ...' messages"
                            )
                            memory_input = gr.Textbox(
                                label="Add a fact to remember",
                                placeholder="Example: I prefer answers with Python examples",
                                lines=2
                            )
                            with gr.Row():
                                add_memory_btn = example_button("➕ Save fact")
                                save_last_btn = example_button("💾 Save last answer")
                            memory_view = gr.Markdown("No saved memory yet.", elem_classes=["pe-help"])
                            memory_delete = gr.Dropdown(
                                label="Delete one memory",
                                choices=[],
                                value=None
                            )
                            with gr.Row():
                                delete_memory_btn = example_button("🗑 Delete selected")
                                clear_memory_btn = example_button("🧹 Clear all memory")
                            with gr.Row():
                                ex_mem_add = example_button("1️⃣ Add sample facts")
                                ex_mem_q = example_button("2️⃣ Load recall question")
                            compare_memory_btn = gr.Button(
                                "⚖️ Compare Memory ON vs OFF",
                                variant="primary",
                                elem_classes=["compare-btn"]
                            )

                        # ---- B. Documents + RAG ----
                        with gr.Accordion(
                            "📚 B · Documents + RAG",
                            open=False,
                            elem_classes=["prompt-panel"]
                        ):
                            help_text(
                                "**What:** upload a document. It is cut into chunks, converted to embeddings (free Gemini "
                                "embeddings), the best chunks are retrieved for your question and the AI answers from them with citations like [1].\n\n"
                                "**Example:** a sample company handbook is built in.\n\n"
                                "**Steps**\n"
                                "1. Press **📄 Add sample document** (or upload your own and press **📥 Index uploaded files**).\n"
                                "2. Tick **Use documents (RAG)**.\n"
                                "3. Press **🧪 Q1: leave policy**, then **➤ Send**.\n"
                                "4. Read the answer. It cites [1], [2] and the right panel lists the retrieved chunks with scores.\n"
                                "5. Press **🧪 Q3: not in document**, then Send. It should say the documents do not contain it.\n"
                                "6. Press **⚖️ Compare RAG ON vs OFF** for the same question.\n\n"
                                "**Retrieval quality:** use **🔍 Preview retrieval** to see which chunks are found without calling the chat model."
                            )
                            use_rag = gr.Checkbox(
                                label="Use documents (RAG)",
                                value=False,
                                info="Retrieves the best chunks for every question"
                            )
                            doc_files = gr.File(
                                label="Upload documents (txt, md, csv, pdf, docx)",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".csv", ".pdf", ".docx"],
                                elem_id="doc-files"
                            )
                            chunk_size = gr.Slider(
                                minimum=20, maximum=400, value=80, step=10,
                                label="Chunk size (words)",
                                info="Smaller chunks = more precise, larger = more context"
                            )
                            chunk_overlap = gr.Slider(
                                minimum=0, maximum=100, value=20, step=5,
                                label="Chunk overlap (words)"
                            )
                            top_k = gr.Slider(
                                minimum=1, maximum=8, value=3, step=1,
                                label="Top-K chunks to retrieve"
                            )
                            with gr.Row():
                                index_btn = example_button("📥 Index uploaded files")
                                sample_doc_btn = example_button("📄 Add sample document")
                            clear_index_btn = example_button("🧹 Clear index")
                            rag_status = gr.Markdown("No documents indexed yet.", elem_classes=["pe-help"])

                            with gr.Row():
                                ex_rag_q1 = example_button("🧪 Q1: leave policy")
                                ex_rag_q2 = example_button("🧪 Q2: laptop")
                            ex_rag_q3 = example_button("🧪 Q3: not in document")

                            retrieval_query = gr.Textbox(
                                label="Retrieval test question",
                                placeholder="Filled by the Q buttons, or type your own",
                                lines=1
                            )
                            preview_btn = example_button("🔍 Preview retrieval")
                            retrieval_preview = gr.Markdown("", elem_classes=["pe-help"], elem_id="retrieval-preview")
                            compare_rag_btn = gr.Button(
                                "⚖️ Compare RAG ON vs OFF",
                                variant="primary",
                                elem_classes=["compare-btn"]
                            )

    # ---------- Header tabs ----------
    tab_outputs = [
        settings_column,
        eval_column,
        active_tab,
        context_tab,
        prompt_tab,
        memory_tab,
        context_tab_btn,
        prompt_tab_btn,
        memory_tab_btn,
        eval_tab_btn
    ]

    eval_tab_btn.click(fn=click_eval_tab, inputs=[active_tab], outputs=tab_outputs)
    context_tab_btn.click(fn=click_context_tab, inputs=[active_tab], outputs=tab_outputs)
    prompt_tab_btn.click(fn=click_prompt_tab, inputs=[active_tab], outputs=tab_outputs)
    memory_tab_btn.click(fn=click_memory_tab, inputs=[active_tab], outputs=tab_outputs)

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
        format_hint,
        model_dropdown,
        eval_log,
        use_memory,
        use_rag,
        top_k,
        rag_index
    ]

    chat_outputs = [
        chat_display,
        chat_history,
        message,
        evaluation_display,
        eval_log
    ]

    plot_inputs = [eval_log, metric_radio]

    memory_outputs = [memory_input, memory_view, memory_delete]
    memory_list_outputs = [memory_view, memory_delete]

    submit_btn.click(
        fn=chat_with_gemini, inputs=chat_inputs, outputs=chat_outputs
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot
    ).then(fn=memory_ui_update, inputs=[], outputs=memory_list_outputs)

    message.submit(
        fn=chat_with_gemini, inputs=chat_inputs, outputs=chat_outputs
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot
    ).then(fn=memory_ui_update, inputs=[], outputs=memory_list_outputs)

    # ---------- Model dropdown + graph metric ----------
    refresh_models_btn.click(
        fn=refresh_models, inputs=[model_dropdown], outputs=[model_dropdown, model_status]
    )
    default_model_btn.click(fn=use_default_model, inputs=[], outputs=[model_dropdown])
    metric_radio.change(fn=update_plot, inputs=plot_inputs, outputs=eval_plot)

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
        compare_techniques,
        model_dropdown,
        eval_log
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
        eval_tab_btn,
        memory_tab_btn,
        eval_log,
        eval_column
    ]

    compare_btn.click(
        fn=compare_prompts, inputs=compare_inputs, outputs=compare_outputs
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot)

    # ---------- Memory ON/OFF and RAG ON/OFF comparisons ----------
    context_compare_inputs = [
        message, chat_history, active_tab, temperature, max_output_tokens,
        top_p, model_dropdown, eval_log, rag_index, top_k
    ]
    compare_memory_btn.click(
        fn=compare_memory, inputs=context_compare_inputs, outputs=compare_outputs
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot)
    compare_rag_btn.click(
        fn=compare_rag, inputs=context_compare_inputs, outputs=compare_outputs
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot)

    # ---------- Memory tab ----------
    add_memory_btn.click(fn=ui_add_memory, inputs=[memory_input], outputs=memory_outputs)
    memory_input.submit(fn=ui_add_memory, inputs=[memory_input], outputs=memory_outputs)
    save_last_btn.click(fn=ui_save_last_answer, inputs=[chat_history], outputs=memory_outputs)
    delete_memory_btn.click(fn=ui_delete_memory, inputs=[memory_delete], outputs=memory_outputs)
    clear_memory_btn.click(fn=ui_clear_memory, inputs=[], outputs=memory_outputs)
    ex_mem_add.click(fn=ui_add_sample_memory, inputs=[], outputs=memory_outputs)
    ex_mem_q.click(fn=make_filler(SAMPLE_MEMORY_QUESTION), inputs=[], outputs=[message])

    # ---------- Documents + RAG ----------
    index_btn.click(
        fn=ui_index_files, inputs=[doc_files, chunk_size, chunk_overlap, rag_index],
        outputs=[rag_index, rag_status]
    )
    sample_doc_btn.click(
        fn=ui_add_sample_doc, inputs=[chunk_size, chunk_overlap, rag_index],
        outputs=[rag_index, rag_status]
    )
    clear_index_btn.click(fn=ui_clear_index, inputs=[], outputs=[rag_index, rag_status])
    preview_btn.click(
        fn=ui_preview_retrieval, inputs=[rag_index, retrieval_query, top_k],
        outputs=[retrieval_preview]
    )
    for button, question in (
        (ex_rag_q1, SAMPLE_RAG_Q1), (ex_rag_q2, SAMPLE_RAG_Q2), (ex_rag_q3, SAMPLE_RAG_Q3)
    ):
        button.click(fn=make_filler(question, question), inputs=[], outputs=[message, retrieval_query])

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chat_display,
            chat_history,
            message,
            evaluation_display,
            eval_log
        ]
    ).then(fn=update_plot, inputs=plot_inputs, outputs=eval_plot)

    demo.load(fn=None, inputs=None, outputs=None, js=autoscroll_js)
    demo.load(fn=refresh_models, inputs=[model_dropdown], outputs=[model_dropdown, model_status])
    demo.load(fn=memory_ui_update, inputs=[], outputs=memory_list_outputs)


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