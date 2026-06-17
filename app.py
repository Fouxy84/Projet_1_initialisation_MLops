import gradio as gr
import json
import joblib
import pandas as pd
import numpy as np
import onnxruntime as rt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# Load global variables
try:
    with open(BASE_DIR / "api" / "inference_pool.json", "r") as f:
        pool = json.load(f)
except FileNotFoundError:
    with open(BASE_DIR / "inference_pool.json", "r") as f:
        pool = json.load(f)

# Ensure valid list for dropdown
client_indices = [str(item.get("Client_index", item.get("client_id"))) for item in pool]

# Map dictionaries for quick lookup
pool_dict = {str(item.get("Client_index", item.get("client_id"))): item["features"] for item in pool}

# Load artifacts for models
models_info = {}
for flavor in ["XGBoost", "LightGBM"]:
    onnx_path = ARTIFACTS_DIR / f"{flavor}.onnx"
    features_path = ARTIFACTS_DIR / f"{flavor}_features.joblib"
    threshold_path = ARTIFACTS_DIR / f"{flavor}_threshold.json"
    
    if onnx_path.exists():
        session = rt.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        features_list = joblib.load(features_path)
        with open(threshold_path, "r") as f:
            t = json.load(f)
            threshold = t.get("threshold", t.get("best_threshold", 0.5))
            
        models_info[flavor] = {
            "session": session,
            "input_name": input_name,
            "features": features_list,
            "threshold": float(threshold)
        }

def predict_score(client_idx, model_name):
    if client_idx not in pool_dict:
        return f"Client ID '{client_idx}' not found.", "", ""
    
    if model_name not in models_info:
        return f"Model '{model_name}' not loaded.", "", ""
    
    features_data = pool_dict[client_idx]
    info = models_info[model_name]
    
    input_df = pd.DataFrame([features_data])
    input_df = input_df.fillna(0)
    input_df = input_df.reindex(columns=info["features"])
    
    input_array = input_df.to_numpy().astype(np.float32)
    outputs = info["session"].run(None, {info["input_name"]: input_array})
    
    proba = float(outputs[1][0][1])
    threshold = info["threshold"]
    
    decision = "🛑 Rejected (Default Risk)" if proba >= threshold else "✅ Approved (No Default Risk)"
    
    return f"{proba:.4f}", f"{threshold:.4f}", decision

with gr.Blocks(title="HomeCredit Default Risk Scoring") as demo:
    gr.Markdown("# 🏦 HomeCredit Scoring API Demo")
    gr.Markdown("Select a client and a model to calculate the default probability.")
    
    with gr.Row():
        client_dropdown = gr.Dropdown(choices=client_indices[:1000], label="Client Index", value=client_indices[0] if client_indices else None)
        model_dropdown = gr.Dropdown(choices=list(models_info.keys()), label="Model", value=list(models_info.keys())[0] if models_info else None)
    
    btn = gr.Button("🔍 Score Client", variant="primary")
    
    with gr.Row():
        out_proba = gr.Textbox(label="Default Probability")
        out_thresh = gr.Textbox(label="Decision Threshold")
        out_decision = gr.Textbox(label="Credit Decision")
        
    btn.click(fn=predict_score, inputs=[client_dropdown, model_dropdown], outputs=[out_proba, out_thresh, out_decision])

if __name__ == "__main__":
    demo.launch()