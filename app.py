# ============================================
# NEUROGEN AI ASSISTANT
# Python + Google Gemini API + Gradio
# Deployment: Render
# ============================================

import os
import html
import gradio as gr

from google import genai
from dotenv import load_dotenv
from datetime import datetime


# --------------------------------------------
# 1. GEMINI API CONFIGURATION
# --------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing! "
        "Configure it in your local .env file "
        "or Render Environment Variables."
    )

client = genai.Client(api_key=api_key)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

print("Gemini client initialized successfully!")


# --------------------------------------------
# 2. DISPLAY CHAT MESSAGES
# --------------------------------------------


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
                    <div class="bubble-label bot-label">
                        ✨ NEUROGEN AI
                    </div>
                    {content}
                </div>
            </div>
            """

    return f"""
    <div class="chat-scroll">
        {output}
    </div>
    """


# --------------------------------------------
# 3. GEMINI CHATBOT FUNCTION
# --------------------------------------------


def chat_with_gemini(
    message, history, user_name, user_role, skill_level, interests, reference_context
):

    history = history or []

    # Validate message
    if not message or not message.strip():
        yield show_chat(history), history, ""
        return

    message = message.strip()

    # ----------------------------------------
    # USER CONTEXT
    # ----------------------------------------

    user_context = f"""
User Profile:
Name: {user_name or 'Not provided'}
Role: {user_role or 'Not provided'}
Skill Level: {skill_level or 'Beginner'}
Interests: {interests or 'Not provided'}

Instructions:
Personalize your answer according to the user's
role, skill level and interests.
Use simple explanations and suitable examples
based on the user's skill level.
"""

    # ----------------------------------------
    # DYNAMIC CONTEXT
    # ----------------------------------------

    current_datetime = datetime.now().astimezone()

    dynamic_context = f"""
Current Date: {current_datetime.strftime("%d-%m-%Y")}
Current Time: {current_datetime.strftime("%I:%M %p")}
Day: {current_datetime.strftime("%A")}

Instructions:
Use the current date and time when answering
time-related questions.
"""

    # ----------------------------------------
    # GROUNDING CONTEXT
    # ----------------------------------------

    if reference_context and reference_context.strip():

        grounding_context = f"""
        Reference Information:
        {reference_context.strip()}

        Instructions:
        1. Answer using the provided reference information.
        2. Do not invent facts that are not present in the reference.
        3. If the answer is not available in the reference, clearly say:
        "The provided reference does not contain this information."
        """

    else:

        grounding_context = """
        No reference information was provided.

        Answer normally using your general knowledge.
        """

        # ----------------------------------------
        # COMBINE ALL CONTEXT
        # ----------------------------------------

        contextual_message = f"""
        {user_context}

        Dynamic Context:
        {dynamic_context}

        Grounding Context:
        {grounding_context}

        User Question:
        {message}
        """

    # ----------------------------------------
    # CONVERSATION HISTORY
    # ----------------------------------------

    contents = []

    # ----------------------------------------
    # CONTEXT FILTERING
    # ----------------------------------------

    # Keep only the latest 10 messages
    filtered_history = history[-10:]

    # ----------------------------------------
    # CONVERSATION HISTORY
    # ----------------------------------------

    contents = []

    for item in filtered_history:

        role = item["role"]

        if role in ["user", "assistant"]:

            contents.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": str(item["content"])}],
                }
            )

    # Add current question with context
    contents.append({"role": "user", "parts": [{"text": contextual_message}]})

    # Store user question
    base_history = history + [{"role": "user", "content": message}]

    # Display user question
    yield show_chat(base_history), base_history, ""

    # ----------------------------------------
    # GEMINI STREAMING RESPONSE
    # ----------------------------------------

    try:

        answer = ""

        stream = client.models.generate_content_stream(model=MODEL, contents=contents)

        for chunk in stream:

            if chunk.text:

                answer += chunk.text

                current_history = base_history + [
                    {"role": "assistant", "content": answer}
                ]

                yield (show_chat(current_history), current_history, "")

        if not answer:
            answer = "No response received. Please try again."

        # Final history
        final_history = base_history + [{"role": "assistant", "content": answer}]

        yield (show_chat(final_history), final_history, "")

    except Exception as e:

        print("Gemini Error:", str(e))

        error_history = base_history + [
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]

        yield (show_chat(error_history), error_history, "")


# --------------------------------------------
# 4. CLEAR CONVERSATION
# --------------------------------------------


def clear_chat():

    return show_chat([]), [], ""


# --------------------------------------------
# 5. PROFESSIONAL UI CSS
# --------------------------------------------

custom_css = """

.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    padding: 20px !important;
}

#main-card {
    border: 1px solid #c9c1e3 !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

#app-heading {
    text-align: center;
    margin-bottom: 12px;
}

#chat-window {
    height: 350px !important;
    min-height: 350px !important;
    max-height: 350px !important;
    overflow: hidden !important;
    padding: 0 !important;
    border: 1.5px solid #c9c1e3 !important;
    border-radius: 15px !important;
    box-sizing: border-box !important;
}

#chat-window .chat-scroll {
    width: 100%;
    height: 100%;
    max-height: 350px;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    box-sizing: border-box;
    padding: 15px;
    scroll-behavior: smooth;
}

#chat-window .chat-scroll::-webkit-scrollbar {
    width: 7px;
}

#chat-window .chat-scroll::-webkit-scrollbar-thumb {
    background: #9885ce;
    border-radius: 10px;
}

.welcome-message {
    min-height: 310px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: var(--body-text-color);
}

.welcome-icon {
    font-size: 42px;
    color: #8b5cf6;
}

.welcome-message h3 {
    font-size: 21px;
    margin: 8px;
}

.welcome-message p {
    margin: 5px;
    font-size: 15px;
}

.welcome-message span {
    font-size: 12px;
    opacity: 0.7;
}

.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}

.user-bubble {
    max-width: 78%;
    padding: 12px 16px;
    background: #e9ddff;
    color: #35215d;
    border-radius: 15px 15px 3px 15px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.bot-row {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}

.bot-bubble {
    width: fit-content;
    max-width: 90%;
    padding: 13px 16px;
    background: #f0eff5;
    color: #252238;
    border: 1px solid #ded9eb;
    border-radius: 15px 15px 15px 3px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.bubble-label {
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.bot-label {
    color: #7654d8;
}

#message-input textarea {
    min-height: 58px !important;
    max-height: 100px !important;
    border: 1.5px solid #c9c1e3 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

#input-row {
    align-items: stretch !important;
    gap: 12px !important;
    margin-top: 14px !important;
}

#send-button {
    min-width: 125px !important;
    min-height: 58px !important;
    border-radius: 12px !important;
    font-weight: bold !important;
}

#clear-button {
    width: 100% !important;
    min-height: 42px !important;
    border-radius: 10px !important;
}

#footer-text {
    text-align: center;
    font-size: 12px;
    opacity: 0.65;
    margin-top: 12px;
}

@media (max-width: 600px) {

    .gradio-container {
        padding: 8px !important;
    }

    #main-card {
        padding: 12px !important;
    }

    #chat-window {
        height: 330px !important;
        min-height: 330px !important;
        max-height: 330px !important;
    }

    #chat-window .chat-scroll {
        max-height: 330px;
        padding: 10px;
    }

    .bot-bubble,
    .user-bubble {
        max-width: 94%;
    }

    #send-button {
        min-width: 85px !important;
    }
}
"""


# --------------------------------------------
# 6. CREATE GRADIO CHATBOT INTERFACE
# --------------------------------------------

with gr.Blocks(css=custom_css) as demo:

    with gr.Column(elem_id="main-card"):

        gr.Markdown(
            """
            # ✨ NeuroGen AI Assistant

            ### Your Smart AI Conversation Partner
            """,
            elem_id="app-heading",
        )

        gr.Markdown("🟢 **Online** | Powered by Google Gemini")

        # ------------------------------------
        # USER CONTEXT UI
        # ------------------------------------

        gr.Markdown("### 👤 User Context")

        user_name = gr.Textbox(label="Your Name", placeholder="Example: Ravi")

        user_role = gr.Textbox(
            label="Your Role", placeholder="Example: College Student"
        )

        skill_level = gr.Dropdown(
            label="Skill Level",
            choices=["Beginner", "Intermediate", "Advanced"],
            value="Beginner",
        )

        interests = gr.Textbox(
            label="Your Interests", placeholder="Example: Python, AI, Data Science"
        )

        # ------------------------------------
        # CONVERSATION HISTORY STATE
        # ------------------------------------

        chat_history = gr.State([])

        gr.Markdown("### 📚 Reference Context")

        reference_context = gr.Textbox(
            label="Provide Reference Information",
            placeholder="Paste your document or reference information here...",
            lines=5,
        )

        chat_display = gr.HTML(value=show_chat([]), elem_id="chat-window")

        # ------------------------------------
        # MESSAGE INPUT
        # ------------------------------------

        with gr.Row(elem_id="input-row"):

            message = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                lines=2,
                scale=8,
                elem_id="message-input",
            )

            submit_btn = gr.Button(
                "➤ Submit", variant="primary", scale=2, elem_id="send-button"
            )

        # ------------------------------------
        # CLEAR BUTTON
        # ------------------------------------

        clear_btn = gr.Button(
            "🗑️ Clear Conversation", variant="secondary", elem_id="clear-button"
        )

        gr.Markdown(
            "Designed with Gradio | Powered by Gemini AI", elem_id="footer-text"
        )

    # ----------------------------------------
    # SUBMIT BUTTON EVENT
    # ----------------------------------------

    submit_btn.click(
        fn=chat_with_gemini,
        inputs=[
            message,
            chat_history,
            user_name,
            user_role,
            skill_level,
            interests,
            reference_context,
        ],
        outputs=[chat_display, chat_history, message],
    )

    # ----------------------------------------
    # ENTER KEY EVENT
    # ----------------------------------------

    message.submit(
        fn=chat_with_gemini,
        inputs=[
            message,
            chat_history,
            user_name,
            user_role,
            skill_level,
            interests,
            reference_context,
        ],
        outputs=[chat_display, chat_history, message],
    )

    # ----------------------------------------
    # CLEAR CONVERSATION EVENT
    # ----------------------------------------

    clear_btn.click(
        fn=clear_chat, inputs=[], outputs=[chat_display, chat_history, message]
    )


# --------------------------------------------
# 7. START APPLICATION
# --------------------------------------------

if __name__ == "__main__":

    IS_LOCAL = False  # True = Local, False = Render Live

    port = int(os.environ.get("PORT", 7860))

    host = "127.0.0.1" if IS_LOCAL else "0.0.0.0"

    print(f"Starting NeuroGen AI on {host}:{port}")

    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=False
    )
