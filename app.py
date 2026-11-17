# ============================================
# NEUROGEN AI ASSISTANT
# Python + Google Gemini API + Gradio
# Compact one-screen layout | Local + Render
# ============================================

import os
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
# 2. DISPLAY CHAT MESSAGES
# ============================================

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
        content = html.escape(str(item["content"])).replace("\n", "<br>")

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
# 3. EVALUATION REPORT
# ============================================

def create_evaluation_report(
    latency, prompt_tokens, response_tokens, reference_provided
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

    return f"""
## 📊 Chatbot Evaluation

### Automatic Performance Metrics

| Metric | Result |
|---|---|
| Latency | {latency:.2f} seconds |
| Input Tokens | {prompt_value} |
| Output Tokens | {response_value} |
| Total Tokens | {total_tokens} |

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

*Quality ratings require human review. They are not automatically verified.*
"""


# ============================================
# 4. GEMINI CHATBOT FUNCTION
# ============================================

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
    top_p
):
    history = history or []

    if not message or not message.strip():
        yield show_chat(history), history, "", "Please enter a question."
        return

    message = message.strip()

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

    reference_provided = bool(reference_context and reference_context.strip())

    if reference_provided:
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

    contextual_message = f"""
{user_context}
Dynamic Context:
{dynamic_context}
Grounding Context:
{grounding_context}
User Question:
{message}
"""

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
            config=genai.types.GenerateContentConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
                top_p=float(top_p)
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
        final_history = base_history + [{"role": "assistant", "content": answer}]

        evaluation_report = create_evaluation_report(
            latency, prompt_tokens, response_tokens, reference_provided
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
# 5. CLEAR CONVERSATION
# ============================================

def clear_chat():
    return (
        show_chat([]),
        [],
        "",
        "## 📊 Chatbot Evaluation\n\nEvaluation results will appear here."
    )


# ============================================
# 6. SETTINGS OPEN / CLOSE
# ============================================

def toggle_settings(is_open):
    new_state = not bool(is_open)
    return (
        gr.update(visible=new_state),
        gr.update(visible=not new_state),
        new_state
    )


def close_settings():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        False
    )


# ============================================
# 7. COMPACT ONE-SCREEN CSS
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

/* Main split: chat left, evaluation right */
#workspace {
    display: flex !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    align-items: stretch !important;
    gap: 9px !important;
    overflow: hidden !important;
    margin: 0 !important;
}
#chat-column {
    min-width: 0 !important;
    min-height: 0 !important;
    flex: 7 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 7px !important;
    overflow: hidden !important;
    background: linear-gradient(160deg, rgba(16,28,54,.96), rgba(9,15,31,.98)) !important;
    border: 1px solid rgba(62,139,255,.42) !important;
    border-radius: 13px !important;
    padding: 9px !important;
    box-shadow: inset 0 0 22px rgba(53,216,255,.025), 0 0 15px rgba(37,99,235,.06) !important;
}
#right-column {
    min-width: 245px !important;
    width: 30% !important;
    max-width: 310px !important;
    flex: 3 1 0% !important;
    min-height: 0 !important;
    overflow: hidden !important;
    gap: 0 !important;
}
#chat-heading h2 {
    color: #C5EFFF !important;
    font-size: 14px !important;
    margin: 0 !important;
    letter-spacing: .2px !important;
}

/* Inner chat scrolling */
#chat-window {
    flex: 1 1 auto !important;
    min-height: 100px !important;
    height: auto !important;
    max-height: none !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    background: radial-gradient(ellipse at 50% 0%, rgba(31,54,98,.28), transparent 58%), #070B18 !important;
    border: 1px solid rgba(65,137,255,.34) !important;
    border-radius: 10px !important;
    padding: 0 !important;
}
#chat-window .chat-scroll {
    width: 100%; height: 100%; min-height: 0;
    overflow-y: auto !important; overflow-x: hidden !important;
    padding: 10px; box-sizing: border-box; scroll-behavior: smooth;
}
#chat-window .chat-scroll::-webkit-scrollbar { width: 5px; }
#chat-window .chat-scroll::-webkit-scrollbar-thumb { background: linear-gradient(#35D8FF,#855CFF); border-radius: 10px; }
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
#input-row { flex: 0 0 auto !important; gap: 7px !important; align-items: stretch !important; margin: 0 !important; }
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
    flex: 0 0 auto !important; width: 100% !important; min-height: 31px !important;
    background: rgba(23,31,57,.96) !important; color: #DDEBFF !important;
    border: 1px solid rgba(103,132,194,.38) !important; border-radius: 8px !important;
    font-size: 10px !important; margin: 0 !important;
}
#clear-button:hover { background: rgba(47,48,91,.98) !important; border-color: #8B70F4 !important; }

/* Evaluation panel */
#evaluation-panel {
    width: 100% !important; height: 100% !important; min-height: 0 !important;
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
#evaluation-panel::-webkit-scrollbar, #settings-column::-webkit-scrollbar { width: 5px; }
#evaluation-panel::-webkit-scrollbar-thumb, #settings-column::-webkit-scrollbar-thumb { background: linear-gradient(#35D8FF,#855CFF); border-radius: 10px; }

/* Settings occupy the right column in normal layout flow.
   They never float over or cover the chat conversation. */
#right-column {
    display: flex !important;
    flex-direction: column !important;
    min-width: 245px !important;
    width: 30% !important;
    max-width: 330px !important;
    flex: 3 1 0% !important;
    min-height: 0 !important;
    overflow: hidden !important;
    gap: 0 !important;
}
#settings-column {
    position: relative !important;
    inset: auto !important;
    z-index: 1 !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
    flex: 1 1 auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
    background: linear-gradient(150deg, #121D39, #090F21) !important;
    border: 1px solid rgba(68,211,255,.72) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 0 0 1px rgba(168,85,247,.12), 0 0 18px rgba(53,216,255,.12) !important;
}
#settings-header { align-items: center !important; justify-content: space-between !important; gap: 8px !important; }
#sidebar-heading h2 { color: #D8F7FF !important; font-size: 16px !important; margin: 0 !important; }
#settings-description { color: #AAB9D8 !important; font-size: 10px !important; }
#close-settings {
    min-width: 32px !important; min-height: 30px !important;
    background: #202A4B !important; color: #FFFFFF !important;
    border: 1px solid rgba(117,162,255,.5) !important; border-radius: 7px !important;
}
#settings-column .user-panel, #settings-column .llm-panel, #settings-column .reference-panel {
    width: 100% !important; background: #101A32 !important;
    border: 1px solid rgba(101,139,220,.42) !important; border-radius: 9px !important;
    margin: 7px 0 !important; overflow: hidden !important;
}
#settings-column .user-panel summary, #settings-column .llm-panel summary,
#settings-column .reference-panel summary, #settings-column .user-panel button,
#settings-column .llm-panel button, #settings-column .reference-panel button {
    color: #F5F8FF !important; background: linear-gradient(100deg,#192C50,#24204B) !important;
    font-size: 12px !important; font-weight: 600 !important;
}
/* Each expanded accordion gets its own internal scrollbar.
   The Settings panel itself stays fixed; only the open section scrolls. */
#settings-column {
    overflow: hidden !important;
}
#settings-column .user-panel .wrap,
#settings-column .llm-panel .wrap,
#settings-column .reference-panel .wrap {
    background: #101A32 !important;
    color: #FFFFFF !important;
    padding: 9px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    scrollbar-width: thin !important;
    scrollbar-color: #35D8FF #101A32 !important;
}
#settings-column .user-panel .wrap {
    max-height: 225px !important;
}
#settings-column .llm-panel .wrap {
    max-height: 205px !important;
}
#settings-column .reference-panel .wrap {
    max-height: 190px !important;
}
#settings-column .user-panel .wrap::-webkit-scrollbar,
#settings-column .llm-panel .wrap::-webkit-scrollbar,
#settings-column .reference-panel .wrap::-webkit-scrollbar {
    width: 5px !important;
}
#settings-column .user-panel .wrap::-webkit-scrollbar-thumb,
#settings-column .llm-panel .wrap::-webkit-scrollbar-thumb,
#settings-column .reference-panel .wrap::-webkit-scrollbar-thumb {
    background: linear-gradient(#35D8FF, #855CFF) !important;
    border-radius: 10px !important;
}
#settings-column input:not([type="checkbox"]), #settings-column textarea, #settings-column select {
    background: #080D1B !important; color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important; border: 1px solid rgba(89,156,255,.48) !important;
    border-radius: 7px !important; font-size: 12px !important; box-shadow: none !important;
}
#settings-column input::placeholder, #settings-column textarea::placeholder { color: #9EAFCE !important; -webkit-text-fill-color: #9EAFCE !important; opacity: 1 !important; }
#settings-column input:focus, #settings-column textarea:focus { border-color: #35D8FF !important; box-shadow: 0 0 10px rgba(53,216,255,.12) !important; }
#settings-column label, #settings-column .block-label { color: #F4F7FF !important; background: transparent !important; font-size: 11px !important; }
#settings-column .info, #settings-column small { color: #AAB9D8 !important; font-size: 9px !important; }
#settings-column input[type="checkbox"], #settings-column input[type="range"] { accent-color: #35D8FF !important; }
#settings-column [role="listbox"], #settings-column [role="option"] { background: #101A32 !important; color: #FFFFFF !important; }

#footer-text { flex: 0 0 auto !important; text-align: center !important; color: #8395B8 !important; font-size: 8px !important; margin: 0 !important; }

/* Mobile: compact viewport, internal workspace scroll */
@media (max-width: 760px) {
    .gradio-container { padding: 5px !important; }
    #main-card { height: calc(100vh - 10px) !important; max-height: calc(100vh - 10px) !important; padding: 7px !important; gap: 5px !important; }
    #app-heading h1 { font-size: 16px !important; }
    #app-heading h3 { font-size: 9px !important; }
    .settings-blink { display: none !important; }
    #workspace { flex-direction: column !important; overflow-y: auto !important; overflow-x: hidden !important; }
    #chat-column { flex: 0 0 420px !important; width: 100% !important; min-height: 420px !important; }
    #right-column { flex: 0 0 240px !important; width: 100% !important; max-width: 100% !important; min-width: 0 !important; min-height: 240px !important; }
    #evaluation-panel { min-height: 220px !important; }
    #settings-column { width: 100% !important; max-width: 100% !important; min-height: 380px !important; }
}

/* =====================================================
   INDEPENDENT INLINE SCROLL AREAS — FINAL OVERRIDES
   1) Chat history scrolls; message composer remains fixed.
   2) User Context body scrolls below its fixed header.
   3) LLM Parameters body scrolls below its fixed header.
   4) Reference Context also has its own body scrollbar.
   ===================================================== */
html, body, .gradio-container, #main-card {
    overflow: hidden !important;
    overscroll-behavior: none !important;
}
#main-card, #workspace, #chat-column, #right-column {
    min-height: 0 !important;
    max-height: 100% !important;
}
#workspace { overflow: hidden !important; }
#chat-column { display: flex !important; flex-direction: column !important; }
/* Chatbot History: constrain every Gradio wrapper, then scroll only the messages. */
#chat-column {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
#chat-heading { flex: 0 0 auto !important; }
#chat-window {
    display: flex !important;
    flex: 1 1 0% !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: 0 !important;
    position: relative !important;
}
/* Gradio HTML component wrappers must also be shrinkable. */
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
    pointer-events: auto !important;
    scrollbar-gutter: stable !important;
    scrollbar-width: thin !important;
    scrollbar-color: #35D8FF #101A32 !important;
    padding: 10px !important;
    box-sizing: border-box !important;
}
#chat-window .chat-scroll::-webkit-scrollbar {
    width: 8px !important;
    display: block !important;
}
#chat-window .chat-scroll::-webkit-scrollbar-track {
    background: #101A32 !important;
    border-left: 1px solid rgba(53,216,255,.15) !important;
    border-radius: 10px !important;
}
#chat-window .chat-scroll::-webkit-scrollbar-thumb {
    min-height: 35px !important;
    background: linear-gradient(180deg, #35D8FF, #855CFF) !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 10px !important;
}
#chat-window .chat-scroll::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #75ECFF, #A77BFF) !important;
}
#input-row, #clear-button { flex-shrink: 0 !important; }
#right-column { display: flex !important; flex-direction: column !important; }
#evaluation-panel, #settings-column {
    flex: 1 1 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 100% !important;
}
#settings-column { overflow: hidden !important; }
#settings-column .user-panel,
#settings-column .llm-panel,
#settings-column .reference-panel {
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    overflow: hidden !important;
    flex: 0 0 auto !important;
}
#settings-column .user-panel > summary,
#settings-column .llm-panel > summary,
#settings-column .reference-panel > summary,
#settings-column .user-panel > button,
#settings-column .llm-panel > button,
#settings-column .reference-panel > button {
    flex: 0 0 auto !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 2 !important;
}
#settings-column .user-panel .wrap,
#settings-column .llm-panel .wrap,
#settings-column .reference-panel .wrap {
    min-height: 0 !important;
    max-height: 190px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
}
#settings-column .user-panel .wrap { max-height: 190px !important; }
#settings-column .llm-panel .wrap { max-height: 175px !important; }
#settings-column .reference-panel .wrap { max-height: 155px !important; }
#settings-column .user-panel .wrap::-webkit-scrollbar,
#settings-column .llm-panel .wrap::-webkit-scrollbar,
#settings-column .reference-panel .wrap::-webkit-scrollbar { width: 5px !important; }
#settings-column .user-panel .wrap::-webkit-scrollbar-thumb,
#settings-column .llm-panel .wrap::-webkit-scrollbar-thumb,
#settings-column .reference-panel .wrap::-webkit-scrollbar-thumb {
    background: linear-gradient(#35D8FF, #855CFF) !important;
    border-radius: 10px !important;
}

@media (max-width: 760px) {
    .gradio-container { padding: 5px !important; }
    #main-card { height: calc(100dvh - 10px) !important; max-height: calc(100dvh - 10px) !important; }
    #workspace {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        overflow: hidden !important;
        gap: 6px !important;
    }
    #chat-column {
        flex: 1 1 0 !important;
        width: 100% !important;
        min-height: 0 !important;
    }
    #right-column {
        flex: 0 0 34% !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
    }
    #evaluation-panel, #settings-column {
        height: 100% !important;
        min-height: 0 !important;
    }
    #settings-column { padding: 7px !important; }
    #settings-column .user-panel .wrap { max-height: 145px !important; }
    #settings-column .llm-panel .wrap { max-height: 135px !important; }
    #settings-column .reference-panel .wrap { max-height: 125px !important; }
    #input-row { gap: 5px !important; }
    #message-input textarea { min-height: 42px !important; }
    #send-button { min-width: 65px !important; }
}

"""


# ============================================
# 8. SINGLE-OPEN ACCORDION JAVASCRIPT
# ============================================

single_open_js = r"""
() => {
    if (window.neurogenScrollUIInstalled) return;
    window.neurogenScrollUIInstalled = true;

    const selectors = [
        { selector: "#settings-column .user-panel", max: "190px" },
        { selector: "#settings-column .llm-panel", max: "175px" },
        { selector: "#settings-column .reference-panel", max: "155px" }
    ];

    function headerFor(root) {
        return root.querySelector(
            ":scope > summary, :scope > button[aria-expanded], " +
            ":scope > [role='button'][aria-expanded], " +
            "summary, button[aria-expanded], [role='button'][aria-expanded]"
        );
    }

    function contentFor(root, header) {
        if (root.tagName === "DETAILS") {
            return Array.from(root.children).find(el => el !== header) || null;
        }
        const explicit = root.querySelector(
            "[data-testid='accordion-content'], .accordion-content, .content"
        );
        if (explicit && explicit !== header && !explicit.contains(header)) return explicit;

        // Gradio versions render the accordion header and body as sibling nodes.
        const sibling = Array.from(root.children).find(
            el => el !== header && !(header && el.contains(header)) &&
                  (el.querySelector("input, textarea, select, [role='slider']") ||
                   el.matches("[data-testid='accordion-content']"))
        );
        if (sibling) return sibling;

        // Fallback: locate the first form control and use its direct child of the panel.
        const control = root.querySelector("input, textarea, select, [role='slider']");
        if (!control) return null;
        let node = control;
        while (node.parentElement && node.parentElement !== root) node = node.parentElement;
        return node !== header ? node : null;
    }

    function setScrollLayout() {
        selectors.forEach(item => {
            document.querySelectorAll(item.selector).forEach(root => {
                // Prefer the native details node when Gradio wraps it in a div.
                const detailsNode = root.matches("details") ? root : root.querySelector("details");
                const panel = detailsNode || root;
                const header = headerFor(panel);
                const body = contentFor(panel, header);
                panel.style.setProperty("display", "flex", "important");
                panel.style.setProperty("flex-direction", "column", "important");
                panel.style.setProperty("min-height", "0", "important");
                panel.style.setProperty("max-height", item.max, "important");
                panel.style.setProperty("overflow", "hidden", "important");
                panel.style.setProperty("box-sizing", "border-box", "important");
                if (header) {
                    header.style.setProperty("flex", "0 0 auto", "important");
                    header.style.setProperty("position", "relative", "important");
                    header.style.setProperty("z-index", "2", "important");
                }
                if (body) {
                    body.style.setProperty("min-height", "0", "important");
                    body.style.setProperty("max-height", item.max, "important");
                    body.style.setProperty("overflow-y", "auto", "important");
                    body.style.setProperty("overflow-x", "hidden", "important");
                    body.style.setProperty("overscroll-behavior", "contain", "important");
                    body.style.setProperty("scrollbar-width", "thin", "important");
                    body.style.setProperty("scrollbar-color", "#35D8FF #101A32", "important");
                }
            });
        });

        // The conversation is its own scroll area; keep the composer outside it.
        const chatRoot = document.querySelector("#chat-window");
        const chat = chatRoot && chatRoot.querySelector(".chat-scroll");
        if (chatRoot) {
            chatRoot.style.setProperty("display", "flex", "important");
            chatRoot.style.setProperty("flex", "1 1 0%", "important");
            chatRoot.style.setProperty("height", "0", "important");
            chatRoot.style.setProperty("min-height", "0", "important");
        }
        if (chat) {
            chat.style.setProperty("display", "block", "important");
            chat.style.setProperty("flex", "1 1 0%", "important");
            chat.style.setProperty("height", "0", "important");
            chat.style.setProperty("min-height", "0", "important");
            chat.style.setProperty("overflow-y", "auto", "important");
            chat.style.setProperty("overflow-x", "hidden", "important");
            chat.style.setProperty("touch-action", "pan-y", "important");
            requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
        }
    }

    function closeOtherPanels(current) {
        const details = current.querySelector("details");
        const expanded = current.querySelector("button[aria-expanded='true'], [role='button'][aria-expanded='true']");
        const currentOpen = (details && details.open) || !!expanded;
        if (!currentOpen) return;
        selectors.forEach(item => {
            document.querySelectorAll(item.selector).forEach(other => {
                if (other === current) return;
                const otherDetails = other.querySelector("details");
                if (otherDetails && otherDetails.open) otherDetails.open = false;
                const otherHeader = other.querySelector("button[aria-expanded='true'], [role='button'][aria-expanded='true']");
                if (otherHeader) otherHeader.click();
            });
        });
    }

    document.addEventListener("click", event => {
        const target = event.target instanceof Element ? event.target : null;
        if (!target) return;
        const root = target.closest(
            "#settings-column .user-panel, #settings-column .llm-panel, #settings-column .reference-panel"
        );
        if (root) setTimeout(() => { closeOtherPanels(root); setScrollLayout(); }, 100);
    }, true);

    // Gradio updates accordion DOM after rendering and after visibility changes.
    const observer = new MutationObserver(() => {
        setScrollLayout();
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-expanded", "open"] });

    // Auto-follow the newest message without moving the page itself.
    const chatRoot = document.querySelector("#chat-window");
    if (chatRoot) {
        const chatObserver = new MutationObserver(() => {
            const scroller = document.querySelector("#chat-window .chat-scroll");
            if (scroller) requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; });
        });
        chatObserver.observe(chatRoot, { childList: true, subtree: true, characterData: true });
    }

    setScrollLayout();
    setTimeout(setScrollLayout, 250);
    setTimeout(setScrollLayout, 900);
}
"""


# ============================================
# 9. CREATE GRADIO INTERFACE
# ============================================

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")
) as demo:

    with gr.Column(elem_id="main-card"):

        # Compact header: title left, settings icon right
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

            with gr.Row(elem_id="settings-control"):
                gr.HTML('<span class="settings-blink">Settings</span>')
                settings_btn = gr.Button(
                    "⚙",
                    size="sm",
                    variant="secondary",
                    elem_id="settings-button"
                )

        settings_state = gr.State(False)

        # Main workspace: chat left, evaluation right
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
                        placeholder="Type your message here...",
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

                        # Settings replace the evaluation panel in the right column.
                        # This is an in-flow layout: it never overlaps the chat.
                with gr.Column(visible=False, elem_id="settings-column") as settings_column:

                    with gr.Row(elem_id="settings-header"):
                        gr.Markdown("## ⚙️ Settings", elem_id="sidebar-heading")
                        close_btn = gr.Button(
                            "✕",
                            size="sm",
                            variant="secondary",
                            elem_id="close-settings"
                        )

                    gr.Markdown(
                        "Manage your AI assistant preferences",
                        elem_id="settings-description"
                    )

                    with gr.Accordion(
                        "👤 User Context",
                        open=False,
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

                    gr.Markdown("NeuroGen AI Settings")

        gr.Markdown(
            "Designed with Gradio | Powered by Gemini AI",
            elem_id="footer-text"
        )

    # Settings open / close
    settings_btn.click(
        fn=toggle_settings,
        inputs=[settings_state],
        outputs=[settings_column, evaluation_display, settings_state]
    )

    close_btn.click(
        fn=close_settings,
        inputs=[],
        outputs=[settings_column, evaluation_display, settings_state]
    )

    # Chat inputs / outputs
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
        top_p
    ]

    chat_outputs = [
        chat_display,
        chat_history,
        message,
        evaluation_display
    ]

    submit_btn.click(
        fn=chat_with_gemini,
        inputs=chat_inputs,
        outputs=chat_outputs
    )

    message.submit(
        fn=chat_with_gemini,
        inputs=chat_inputs,
        outputs=chat_outputs
    )

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

    demo.load(fn=None, inputs=None, outputs=None, js=single_open_js)


# ============================================
# 10. START APPLICATION
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
