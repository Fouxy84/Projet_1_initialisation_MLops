
import gradio as gr
from fastapi import FastAPI
import requests
import os

<<<<<<< HEAD
<<<<<<< HEAD
GITHUB_TOKEN = os.getenv("REMOVED")
=======
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
>>>>>>> 8f421e6 (update dashbord token access)
REPO = "Fouxy84/Projet_1_initialisation_MLops"
=======
# ==========================
# FASTAPI BACKEND
# ==========================
api = FastAPI()
>>>>>>> d3dc82c (update dashbord app)

@api.get("/")
def greet_json():
    return {"Hello": "World!"}


# ==========================
# GRADIO FRONTEND
# ==========================
def call_api():
     return greet_json()


with gr.Blocks() as demo:
    gr.Markdown("# 🚀 FastAPI + Gradio Demo")

    btn = gr.Button("Call FastAPI")

    output = gr.JSON()

    btn.click(call_api, outputs=output)



app = gr.mount_gradio_app(api, demo, path="/")