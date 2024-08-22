from fastapi import FastAPI, UploadFile
import gradio as gr
from rag.schemas.schemas import ModeEnum, Question, RetrieverSchema, StrategyEnum
from pydantic import BaseModel
import rag.routers.api as api
import re
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage


class AskRequest(BaseModel):
    question: Question
    retrieval_schema: RetrieverSchema


# Function to handle document upload
async def upload_file(file):
    if file is not None:
        # Convert NamedString to BytesIO for file-like behavior
        file_name = re.split(r"\\|\/", file.name)[-1]
        with open(file, "rb") as f:
            response = await api.upload_to_database(
                file=UploadFile(file=f, filename=file_name)
            )
            return response

    # return {"error": "No file uploaded"}


# Function to handle question and answer
def format_history(history):
    format_history = []
    for pair in history:
        human = HumanMessage(content=pair[0])
        system = AIMessage(content=pair[1])
        format_history.append(human)
        format_history.append(system)
    return format_history


async def ask_question(history, question, mode, strategy):
    try:
        response = await api.ask(
            question=Question(question=question),
            retrieval_schema=RetrieverSchema(mode=mode),
            strategy= StrategyEnum(value= strategy),
            history=format_history(history),
        )

        answer = response
        # Append the new question and answer to the chat history
        history.append((question, answer))
    except Exception as e:
        return history, f"Failed to generate answer: {str(e)}"
    return history, ""


def clear_chat(history):
    # Clear the chat history
    return []


# Create Gradio interface
def create_app():
    with gr.Blocks(title="RAG Chatbot") as demo:
        with gr.Row():
            with gr.Column(scale=5, min_width=800):
                gr.Markdown("## Chatbot")
                chatbot = gr.Chatbot(label="Chat Interface")
                with gr.Row():
                    question_input = gr.Textbox(label="Ask a question", scale=6)
                    send_button = gr.Button(
                        "🚀",
                        scale=1,
                        elem_id="send_button",
                    )
                with gr.Row():
                    mode_input = gr.Dropdown(
                        label="Select Mode",
                        choices=[mode.value for mode in ModeEnum],
                        value="default",
                        multiselect=True,
                    )
                    strategy = gr.Dropdown(
                        label="Select Strategy",
                        choices=[st.value for st in StrategyEnum],
                        value="adaptive-rag",
                    )

                clear_button = gr.Button("❌Clear Chat")

                # The `state` argument keeps track of the chat history
                send_button.click(
                    ask_question,
                    inputs=[chatbot, question_input, mode_input, strategy],
                    outputs=[chatbot, question_input],
                )
                clear_button.click(clear_chat, inputs=chatbot, outputs=chatbot)
            with gr.Column(scale=1):
                gr.Markdown("## Upload Document")
                file_input = gr.File(
                    label="Upload your document", file_types=[".pdf", ".txt", ".docx"]
                )
                upload_button = gr.Button("⬆️ Upload ⬆️")
                upload_output = gr.Textbox(label="Upload Status")
                upload_button.click(
                    upload_file, inputs=file_input, outputs=upload_output
                )
            with gr.Row():
                # Tạo văn bản dọc "Created by longduongbao29"
                gr.Markdown(
                    """
                <div style='font-size = 20px;position: absolute; bottom: 0; right: 0;'>
                    🌟Created by longduongbao29🌟
                </div>
                """
                )
    demo.css = """
    #send_button {
        font-size: 40px;
    }
    """
    return demo


# Launch the app
# ui_app = create_app()
# ui_app.launch(server_port=1234, share=True)  # Set share=True to create a public link
ui_app = FastAPI()
favicon_path = "static/chatbot.ico"
ui_app.mount("/static", StaticFiles(directory="static"), name="static")


@ui_app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


ui_app = gr.mount_gradio_app(ui_app, create_app(), path="/")
