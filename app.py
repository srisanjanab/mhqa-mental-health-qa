import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL_PATH = "bsrisanjana/mhqa-bert-finetuned"
MAX_LEN = 128

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!")

EXAMPLES = {
    "Anxiety": ["Which therapy is most effective for generalized anxiety disorder?", "Cognitive Behavioral Therapy (CBT)", "Electroconvulsive Therapy", "Hypnotherapy", "Recreational therapy"],
    "Depression": ["What is a common first-line treatment for major depressive disorder?", "Antipsychotics", "SSRIs (e.g. Fluoxetine)", "Opioids", "Antihistamines"],
    "Trauma": ["Which symptom is characteristic of PTSD?", "Hypermania", "Flashbacks and intrusive memories", "Gradual memory improvement", "Increased sociability"],
    "Obsessive/Compulsive": ["What is the primary behavioral treatment for OCD?", "Systematic desensitization", "Aversion therapy", "Exposure and Response Prevention (ERP)", "Psychoanalysis"],
}

def predict(question, opt1, opt2, opt3, opt4):
    if not question.strip() or any(o.strip() == "" for o in [opt1, opt2, opt3, opt4]):
        return "Please fill in the question and all 4 options.", "", ""
    options = [opt1, opt2, opt3, opt4]
    encodings = tokenizer([question]*4, options, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=encodings["input_ids"].unsqueeze(0), attention_mask=encodings["attention_mask"].unsqueeze(0), token_type_ids=encodings["token_type_ids"].unsqueeze(0))
    probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
    pred_idx = int(probs.argmax())
    bars = ""
    for i, (opt, prob) in enumerate(zip(options, probs)):
        marker = "👉 " if i == pred_idx else "    "
        bars += f"{marker}Option {i+1}: {'█'*int(prob*100/5)}{'░'*(20-int(prob*100/5))} {prob*100:.1f}%\n"
    return f"✅ Option {pred_idx+1}: {options[pred_idx]}", f"{float(probs[pred_idx])*100:.1f}% confidence", bars

def load_example(topic):
    e = EXAMPLES.get(topic, [""]*5)
    return e[0], e[1], e[2], e[3], e[4]

with gr.Blocks(title="MHQA Mental Health QA") as demo:
    gr.Markdown("# 🧠 MHQA — Mental Health Question Answering\nFine-tuned BERT | Accuracy: 79.31% | Domains: Anxiety · Depression · Trauma · OCD\n> For educational purposes only.")

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(label="Question", placeholder="Enter your mental health question...", lines=3)
            with gr.Row():
                opt1 = gr.Textbox(label="Option 1")
                opt2 = gr.Textbox(label="Option 2")
            with gr.Row():
                opt3 = gr.Textbox(label="Option 3")
                opt4 = gr.Textbox(label="Option 4")
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("🔍 Predict", variant="primary", scale=2)

        with gr.Column(scale=1):
            answer_out = gr.Textbox(label="Predicted Answer", interactive=False)
            confidence_out = gr.Textbox(label="Confidence", interactive=False)
            bars_out = gr.Textbox(label="All Option Scores", interactive=False, lines=6)
            topic_dd = gr.Dropdown(choices=list(EXAMPLES.keys()), label="Load Example Topic", value="Anxiety")
            load_btn = gr.Button("Load Example")

    submit_btn.click(fn=predict, inputs=[question_box, opt1, opt2, opt3, opt4], outputs=[answer_out, confidence_out, bars_out])
    clear_btn.click(fn=lambda: ("","","","","","","",""), outputs=[question_box, opt1, opt2, opt3, opt4, answer_out, confidence_out, bars_out])
    load_btn.click(fn=load_example, inputs=[topic_dd], outputs=[question_box, opt1, opt2, opt3, opt4])
    topic_dd.change(fn=load_example, inputs=[topic_dd], outputs=[question_box, opt1, opt2, opt3, opt4])

demo.launch(server_name="0.0.0.0", server_port=7860)