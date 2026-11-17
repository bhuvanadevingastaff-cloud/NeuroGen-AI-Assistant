
# ============================================
# NEUROGEN AI ASSISTANT
# Python + Google Gemini API + Gradio
# Deployment: Render
# ============================================

import os
import html
import gradio as gr
from google import genai

# --------------------------------------------
# 1. GEMINI API CONFIGURATION
# --------------------------------------------

#Local
'''
from dotenv import load_dotenv

load_dotenv()

# Read API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing! "
        "Configure it in your local .env file "
        "or Render Environment Variables."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Gemini model
MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

print("Gemini client initialized successfully!")
'''

#Live
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing! "
        "Please add it in Render Environment Variables."
    )

client = genai.Client(api_key=api_key)

# Use the model name that works with your Gemini API key.
MODEL = "gemini-3.5-flash-lite"

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

        content = html.escape(
            str(item["content"])
        ).replace("\n", "<br>")

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

def chat_with_gemini(message, history):

    history = history or []

    if not message or not message.strip():
        yield show_chat(history), history, ""
        return

    message = message.strip()

    # Prepare previous conversation
    contents = []

    for item in history:

        role = item["role"]

        if role in ["user", "assistant"]:

            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [
                    {"text": str(item["content"])}
                ]
            })

    # Add current question
    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    base_history = history + [
        {"role": "user", "content": message}
    ]

    # Display user question
    yield show_chat(base_history), base_history, ""

    try:

        answer = ""

        # Generate streaming response
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=contents
        )

        for chunk in stream:

            if chunk.text:

                answer += chunk.text

                current_history = base_history + [
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]

                yield (
                    show_chat(current_history),
                    current_history,
                    ""
                )

        if not answer:
            answer = "No response received. Please try again."

        final_history = base_history + [
            {
                "role": "assistant",
                "content": answer
            }
        ]

        yield (
            show_chat(final_history),
            final_history,
            ""
        )

    except Exception as e:

        print("Gemini Error:", str(e))

        error_history = base_history + [
            {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            }
        ]

        yield (
            show_chat(error_history),
            error_history,
            ""
        )


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

with gr.Blocks() as demo:

    with gr.Column(elem_id="main-card"):

        gr.Markdown(
            """
            # ✨ NeuroGen AI Assistant

            ### Your Smart AI Conversation Partner
            """,
            elem_id="app-heading"
        )

        gr.Markdown(
            "🟢 **Online** | Powered by Google Gemini"
        )

        # Conversation history
        chat_history = gr.State([])

        # Chat display
        chat_display = gr.HTML(
            value=show_chat([]),
            elem_id="chat-window"
        )

        # Input and Submit button
        with gr.Row(elem_id="input-row"):

            message = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                lines=2,
                scale=8,
                elem_id="message-input"
            )

            submit_btn = gr.Button(
                "➤ Submit",
                variant="primary",
                scale=2,
                elem_id="send-button"
            )

        # Clear button
        clear_btn = gr.Button(
            "🗑️ Clear Conversation",
            variant="secondary",
            elem_id="clear-button"
        )

        gr.Markdown(
            "Designed with Gradio | Powered by Gemini AI",
            elem_id="footer-text"
        )

    # Submit button event
    submit_btn.click(
        fn=chat_with_gemini,
        inputs=[message, chat_history],
        outputs=[chat_display, chat_history, message]
    )

    # Enter key event
    message.submit(
        fn=chat_with_gemini,
        inputs=[message, chat_history],
        outputs=[chat_display, chat_history, message]
    )

    # Clear conversation event
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat_display, chat_history, message]
    )


# --------------------------------------------
# 7. START APPLICATION ON RENDER
# --------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    print(f"Starting NeuroGen AI on port {port}")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        css=custom_css
    )
