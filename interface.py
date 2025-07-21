import gradio as gr, asyncio
from conversation import ConversationSystem
from config import OPENAI_API_KEY

call_running = False

system = ConversationSystem(
    openai_api_key=OPENAI_API_KEY,
    model_size="base.en",
    language="en",
    llm_model="gpt-4.1-nano",
    silence_threshold=0.005,
    silence_duration=1.5,
    tts_rate=1.0,
)

def show_call():
    """Hide Start, show Stop."""
    return (gr.Button.update(visible=False),
            gr.Button.update(visible=True))

def hang_up():
    """Flip flag, clean up, swap buttons back."""
    global call_running
    call_running = False
    system.cleanup()
    return (gr.Button.update(visible=True),
            gr.Button.update(visible=False))

async def conversation_loop():
    """Yield (user_text, assistant_text) until user presses Stop."""
    global call_running
    call_running = True
    while call_running:
        assistant_text = await system.start_conversation_turn()
        user_text = system.audio_recorder.get_transcribed_text()
        yield user_text, assistant_text   # <-- streaming happens automatically

CSS = """
#start {width:120px;height:120px;border-radius:60px;font-size:48px;
        line-height:110px;text-align:center;background:#ff4040;color:#fff;border:none;}
#start:hover {background:#ff2020;}
#stop  {width:120px;height:120px;border-radius:12px;font-size:32px;
        line-height:110px;text-align:center;background:#555;color:#fff;border:none;}
#stop:hover {background:#444;}
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("## 🎙️ Voice Assistant – click once to start the call")

    start = gr.Button("🎤", elem_id="start")
    stop  = gr.Button("⏹️", elem_id="stop", visible=False)

    user_box      = gr.Textbox(label="You said", interactive=False, lines=2)
    assistant_box = gr.Textbox(label="Assistant replied", interactive=False, lines=3)

    # 1. press Start → swap buttons
    start.click(show_call, None, [start, stop])
    # 2. same click ALSO launches the async generator; no stream= needed in Gradio 5
    start.click(conversation_loop, None, [user_box, assistant_box])

    # 3. Stop button ends the loop and swaps buttons back
    stop.click(hang_up, None, [start, stop])

if __name__ == "__main__":
    demo.launch()
