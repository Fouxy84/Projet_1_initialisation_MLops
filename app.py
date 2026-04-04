import gradio as gr

def greet(name):
    return f"Hello {name}!!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Simple Demo"
)

if __name__ == "__main__":
    demo.launch()