import gradio as gr
import requests
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
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
    gr.Markdown("# MLOps Control Center")

    btn = gr.Button("Run CI Pipeline")
    output = gr.Textbox()

    btn.click(trigger_ci, outputs=output)

demo.launch()