import gradio as gr
import fake_news_classification as fnc

def get_label(content):
    label_news = fnc.classify_news(content)
    if(label_news[0]==0):
        return "Fake News"
    return "True News"

with gr.Blocks() as demo:
    gr.Markdown("# Fake News Classifier")
    title = gr.Textbox(label="Title")
    content = gr.Textbox(label="Content", lines=5)
    output = gr.Textbox(label="News Label")
    btn = gr.Button("Check News")
    btn.click(get_label, inputs=[content], outputs=output)

demo.launch(share=True)