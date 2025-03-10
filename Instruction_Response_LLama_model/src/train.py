# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:42:31 2025

@author: dhars
"""

import os
import torch
import shutil
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
from peft import LoraConfig
from trl import SFTTrainer

def load_base_model(model_path):
    """Load base model with quantization..."""
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quant_config, device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def load_tokenizer(model_path):
    """Load tokenizer and set padding token..."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_lora_config():
    """Define and return LoRA configuration..."""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

def get_training_arguments():
    """Define and return training arguments..."""
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        gradient_checkpointing=True,
    )

def train_model(base_model_path, dataset_name, output_path):
    """Train the model using LoRA fine-tuning..."""
    model = load_base_model(base_model_path)
    tokenizer = load_tokenizer(base_model_path)
    dataset = load_dataset(dataset_name, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=get_lora_config(),
        tokenizer=tokenizer,
        args=get_training_arguments(),
    )

    trainer.train()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

def compress_model(output_path, zip_path):
    """Compress the fine-tuned model into a ZIP file."""
    shutil.make_archive(zip_path, 'zip', output_path)
    print(f"Model compressed to {zip_path}.zip")

def inference(model_path, tokenizer_path, prompt, output_file):
    """Perform inference using the fine-tuned model and save output."""
    model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    output_text = result[0]['generated_text']
    print(output_text)

    with open(output_file, "w") as f:
        f.write(output_text)
    print(f"Inference output saved to {output_file}")

if __name__ == "__main__":
    base_model_path = "/kaggle/input/new_model_testing/pytorch/default/1"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    output_path = "/kaggle/working/fine_tuned_model"
    zip_path = "/kaggle/working/fine_tuned_model"
    test_sentence = "This movie was amazing!"
    inference_model_path = "/kaggle/input/lama2_model_q_and_a/pytorch/default/1"
    tokenizer_model_path = "NousResearch/Llama-2-7b-chat-hf"
    inference_prompt = "Explain the theory of relativity."
    output_file = "inference_output.txt"

    train_model(base_model_path, dataset_name, output_path)
    compress_model(output_path, zip_path)
    inference(inference_model_path, tokenizer_model_path, inference_prompt, output_file)
