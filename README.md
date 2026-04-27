# 🧠 MHQA — Mental Health Question Answering

> Fine-tuning BERT on the MHQA dataset for multiple-choice mental health question answering, with a live deployed web application.

---

## 👥 Team Details

| Name | Roll Number |
|------|-------------|
| Bhukya Jhansi | 160123748007 |
| Bommareddy Sri Sanjana | 160123748008 |
| Budama Lakshmi Pragna Manasvi | 160123748009 |

### 🎓 Project Guide
**Panigrahi Srikanth**  
Assistant Professor, Department of AI & ML  
**CBIT (Chaitanya Bharathi Institute of Technology), Hyderabad**

---

## 📌 Assignment Info
- **Course**: Generative AI
- **Assignment**: 2 (Extension of Assignment 1)
- **Task**: Fine-tune a Language Model on MHQA dataset and deploy with a user-friendly UI

---

## 🔗 Live Links

| Resource | Link |
|----------|------|
| 🚀 Live Demo | [HuggingFace Spaces](https://huggingface.co/spaces/bsrisanjana/mhqa-mental-health-qa) |
| 🤗 Model | [HuggingFace Hub](https://huggingface.co/bsrisanjana/mhqa-bert-finetuned) |
| 💻 Code | [GitHub Repository](https://github.com/bsrisanjana/mhqa-mental-health-qa) |

---

## 📖 About the Project

Mental health remains a challenging problem worldwide, with issues like depression and anxiety becoming increasingly common. Large Language Models (LLMs) have seen vast application in healthcare, specifically in answering medical questions. However, there is a lack of standard benchmarking datasets for question answering (QA) in mental health.

This project fine-tunes **BERT (bert-base-uncased)** on the **MHQA dataset** — a novel multiple-choice dataset for benchmarking language models on mental health QA tasks.

---

## 📁 Dataset

### MHQA (Mental Health Question Answering)
- **Source**: PubMed abstracts
- **Gold standard**: 2,475 expert-verified instances (MHQA-gold)
- **Pseudo-labeled**: ~56.1k pairs
- **Task**: 4-way multiple choice QA
- **Provider**: [IndiaAI — AiKosh](https://aikosh.indiaai.gov.in/home/datasets/details/bharatgen_mhqa_dataset.html)

### MHQA-b (Extended)
- Extended version with `valid_question` labels
- Filtered to valid questions only before training

### Combined Dataset Statistics
| Property | Value |
|----------|-------|
| Total rows | 58,617 |
| After filtering | ~52,000 |
| Train split | 90% |
| Validation split | 10% |
| Answer choices | 4 |

---

## 🏥 Mental Health Domains Covered

| Domain | Description |
|--------|-------------|
| 😰 Anxiety | Panic disorder, phobias, generalized anxiety |
| 😔 Depression | Major depressive disorder, mood disorders |
| 💔 Trauma | PTSD, stress-related disorders |
| 🔄 OCD | Obsessive-compulsive and related disorders |

### Question Types
- **Factoid** — factual recall questions
- **Diagnostic** — identifying conditions or disorders
- **Prognostic** — predicting treatment outcomes
- **Preventive** — prevention strategies

---

## 🛠️ Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| Language | Python 3.11 |
| Model | bert-base-uncased (HuggingFace Transformers) |
| Fine-tuning | PyTorch + HuggingFace Trainer API |
| Training Platform | Google Colab (T4 GPU) |
| Model Hosting | HuggingFace Hub |
| UI Framework | Gradio 4.44.1 |
| Deployment | HuggingFace Spaces |
| Version Control | GitHub |

---

## 🏗️ Model Architecture

```
Question + Option 1  ─┐
Question + Option 2  ─┤──► BERT Tokenizer ──► BERT Encoder ──► Pooler ──► Linear Head ──► Softmax ──► Predicted Answer
Question + Option 3  ─┤                       (bert-base-uncased)
Question + Option 4  ─┘
```

- For each question, 4 input pairs `[CLS] Question [SEP] Option_i [SEP]` are created
- BERT encodes all 4 pairs simultaneously
- A linear classification head predicts the most likely correct option
- **Total Parameters**: ~110M

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **79.31%** |
| Validation Loss | 0.9244 |
| Random Baseline | 25.00% |
| Training Epochs | 3 |
| Batch Size | 8 |
| Max Sequence Length | 128 tokens |

> The model achieves **79.31% accuracy**, which is **3x better** than random chance (25% for 4-choice QA).

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/bsrisanjana/mhqa-mental-health-qa.git
cd mhqa-mental-health-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser and go to `http://localhost:7860`

---

## 📓 Training

The complete training pipeline is in `MHQA_FineTuning.ipynb`:

1. **Cell 1-2**: Install packages and import libraries
2. **Cell 3-4**: Upload MHQA and MHQA-b datasets
3. **Cell 5-6**: Preprocess and split data
4. **Cell 7-8**: Load tokenizer and create PyTorch Dataset
5. **Cell 9-10**: Load BERT model and configure training
6. **Cell 11**: Train the model (~45-90 mins on T4 GPU)
7. **Cell 12**: Evaluate on validation set
8. **Cell 13**: Test inference on sample questions
9. **Cell 14-15**: Save model and upload to HuggingFace Hub

---

## 🖥️ Application Features

- 📝 Enter any mental health question with 4 answer options
- 🔍 Model predicts the correct answer instantly
- 📊 Shows confidence score for each option with visual bar
- 💡 Load example questions by topic (Anxiety / Depression / Trauma / OCD)
- 🌐 Publicly accessible — no login required
- ⚡ Runs on HuggingFace Spaces (free hosting)

---

## 📂 Project Structure

```
mhqa-mental-health-qa/
├── app.py                    # Gradio web application
├── requirements.txt          # Python dependencies
├── MHQA_FineTuning.ipynb    # Training notebook (Google Colab)
└── README.md                 # Project documentation
```

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding mental health conditions.

---

## 📜 License

This project is developed as part of an academic assignment at CBIT, Hyderabad.  
Dataset credit: [IndiaAI — AiKosh MHQA Dataset](https://aikosh.indiaai.gov.in)
