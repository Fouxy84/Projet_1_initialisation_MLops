import gradio as gr
from fastapi import FastAPI

# ==========================
# FASTAPI BACKEND
# ==========================
api = FastAPI()

@api.get("/hello")
def greet_json():
    return {"Hello": "World!"}


# ==========================
# GRADIO FRONTEND
# ==========================
def call_api():
    # appel direct de la fonction backend
    return greet_json()


with gr.Blocks() as demo:
    gr.Markdown("# 🚀 FastAPI + Gradio Demo")

    btn = gr.Button("Call FastAPI")

    output = gr.JSON()

    btn.click(call_api, outputs=output)


# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(api, demo, path="/")