import gradio as gr
import requests
import os

GITHUB_TOKEN = os.getenv("REMOVED")
REPO = "Fouxy84/Projet_1_initialisation_MLops"

def trigger_ci():
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/ci.yml/dispatches"
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    r = requests.post(url, headers=headers, json={"ref": "main"})
    
    if r.status_code == 204:
        return "CI pipeline launched 🚀"
    return f"Error: {r.text}"

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 FastAPI + Gradio Demo")

    btn = gr.Button("Call FastAPI")

    output = gr.JSON()

    btn.click(call_api, outputs=output)


# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(api, demo, path="/")