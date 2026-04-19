# 🧠 MHQA — Mental Health Question Answering

Fine-tuned BERT model for multiple-choice mental health question answering.

## 🔗 Live Demo
👉 [Click here to try the app](https://huggingface.co/spaces/bsrisanjana/mhqa-mental-health-qa)

## 🤗 Model on Hugging Face
👉 [bsrisanjana/mhqa-bert-finetuned](https://huggingface.co/bsrisanjana/mhqa-bert-finetuned)

## 📊 Results
| Metric | Value |
|--------|-------|
| Model | bert-base-uncased |
| Dataset | MHQA + MHQA-b |
| Training samples | ~52,000 |
| Validation Accuracy | **79.31%** |
| Validation Loss | 0.9244 |

## 📁 Dataset
- **MHQA**: Mental Health Question Answering dataset from [IndiaAI](https://aikosh.indiaai.gov.in)
- **MHQA-b**: Extended version with validity labels
- Combined: 58,617 rows across 4 mental health domains

## 🏥 Domains Covered
- 😰 Anxiety
- 😔 Depression
- 💔 Trauma / PTSD
- 🔄 Obsessive Compulsive Disorder (OCD)

## ❓ Question Types
- Factoid, Diagnostic, Prognostic, Preventive

## 🛠️ Tech Stack
- **Model**: HuggingFace Transformers (bert-base-uncased)
- **Fine-tuning**: PyTorch + HuggingFace Trainer API
- **Training**: Google Colab (T4 GPU)
- **Deployment**: HuggingFace Spaces + Gradio

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## 📓 Training
See `MHQA_FineTuning.ipynb` for the complete training pipeline.

## 📌 Assignment Info
- **Course**: Generative AI — Assignment 2
- **Task**: Fine-tune a Language Model on MHQA dataset
