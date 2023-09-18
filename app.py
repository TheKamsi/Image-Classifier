import gradio as gr 
from transformers import pipeline

model = pipeline(task="image-classification", model="julien-c/hotdog-not-hodog")

def predict(image):
    predictions = model(image)
    return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.Image(label="Upload image", type="filepath"),
    outputs=gr.Label(num_top_classes=2),
    title="Hotdog or not Hotdog"
).launch()