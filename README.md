# Instruction_Response_llama

**Project Title: Instruction based text generation using LLaMA model**

📖 Table of Contents

-Introduction
-Dataset
-Model
-Results
-Training
-Inference
-Future Work

📌 Introduction
This project fine-tunes LLaMA generate response to the provided instruction.

🚀 Features:
✔️ Fine-tuning LLaMA using LoRA for efficiency
✔️ Preprocessing & tokenization
✔️ Training on Instruction-Response dataset
✔️ Evaluation & performance metrics

📚 Dataset

Used Instruction-Response dataset in multiple languages. 

Instruction: 

how long does an American football match REALLY last, if you substract all the downtime?

Response: 

According to the Wall Street Journal, the ball is only in play for an average of 11 minutes during the typical NFL game, out of an average total game length of 3 hours and 12 minutes. 

🧠 Model

Fine-tuned LLaMA-13B using LoRA to reduce memory usage.

🔧 Hyperparameters:

| Hyperparameters                | Value |
|-----------------------|-------|
| Learning Rate   | 2e-4   |
| Batch Size   | 4 |
| Epochs      | 2 |
| Optimizer      | paged_adamw_32bit |

📊 Results

📌 Performance Metrics

| Metric                | Score |
|-----------------------|-------|
| BertScore(Accuracy)   | x     |
| BertScore(F1-Score)   | 83.6% |
| BertScore(F1-precision)| 83.6% |
| BertScore(F1-Recall)   | 83.6% |

