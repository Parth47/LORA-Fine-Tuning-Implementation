# src/train.py

import os
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from src.model_setup import load_model_and_tokenizer, prepare_model_for_lora

# --- Configuration ---
HUGGING_FACE_USER_NAME = "your-hf-username" # Change this to your Hugging Face username
MODEL_OUTPUT_NAME = "LORA_Bloom_SQuAD"
TRAINING_STEPS = 100
OUTPUT_DIR = 'outputs'

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def create_prompt(context, question, answer):
  """Creates a formatted prompt for the model."""
  if len(answer["text"]) < 1:
    answer = "Cannot Find Answer"
  else:
    answer = answer["text"][0]
  prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
  return prompt_template

def main():
    # 1. Load Model and Tokenizer
    print("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # 2. Prepare model for LoRA
    print("Preparing model for LoRA training...")
    model = prepare_model_for_lora(model)

    # 3. Configure LoRA
    print("Configuring LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, config)
    print("Trainable parameters after applying LoRA:")
    print_trainable_parameters(lora_model)

    # 4. Load and prepare dataset
    print("Loading and preparing SQuAD v2 dataset...")
    qa_dataset = load_dataset("squad_v2")
    mapped_qa_dataset = qa_dataset.map(
        lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers']))
    )

    # 5. Set up Trainer
    print("Setting up the trainer...")
    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=mapped_qa_dataset["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=TRAINING_STEPS,
            learning_rate=1e-3,
            fp16=True,
            logging_steps=1,
            output_dir=OUTPUT_DIR,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 6. Train
    print(f"Starting training for {TRAINING_STEPS} steps...")
    lora_model.config.use_cache = False  # Silence warnings
    trainer.train()
    print("Training complete!")

    # 7. (Optional) Push to Hub
    # print("Pushing model to Hugging Face Hub...")
    # from huggingface_hub import notebook_login
    # notebook_login() # You'll need to provide your token
    # lora_model.push_to_hub(f"{HUGGING_FACE_USER_NAME}/{MODEL_OUTPUT_NAME}", use_auth_token=True)
    # print(f"Model pushed to {HUGGING_FACE_USER_NAME}/{MODEL_OUTPUT_NAME}")

if __name__ == "__main__":
    main()
