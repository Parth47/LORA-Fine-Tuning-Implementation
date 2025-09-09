# src/inference.py

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Markdown

# --- Configuration ---
# Path to your trained PEFT model. This will typically be a checkpoint directory.
# Update this path to the correct checkpoint from your 'outputs' folder (e.g., 'outputs/checkpoint-100')
PEFT_MODEL_PATH = "outputs/checkpoint-100"

def make_inference(model, tokenizer, context, question):
  """Generates an answer based on the provided context and question."""
  batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt')

  # Move the batch to the same device as the model
  batch = {k: v.to(model.device) for k, v in batch.items()}

  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=200)

  # Use display for better formatting in notebooks, or print for standard terminals
  result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  print("--- Full Model Output ---")
  print(result)
  print("\n--- Generated Answer ---")
  # Extract only the generated part
  answer_start = result.find("### ANSWER\n") + len("### ANSWER\n")
  print(result[answer_start:].strip())


def main():
    print(f"Loading PEFT config from {PEFT_MODEL_PATH}...")
    config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        load_in_8bit=False,  # Load in full precision for inference
        device_map='auto'
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    print(f"Loading LoRA model from {PEFT_MODEL_PATH}...")
    qa_model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
    qa_model.eval() # Set to evaluation mode

    print("\n✅ Model loaded successfully!")

    # --- Example 1 ---
    print("\n--- Running Inference Example 1 ---")
    context1 = "The Moon orbits Earth at an average distance of 384,400 km (238,900 mi)."
    question1 = "At what distance does the Moon orbit the Earth?"
    make_inference(qa_model, tokenizer, context1, question1)

    # --- Example 2 ---
    print("\n--- Running Inference Example 2 ---")
    context2 = "My name is Parth Suresh Suryawanshi and I live in a city called Nashik."
    question2 = "What is my middle name?"
    make_inference(qa_model, tokenizer, context2, question2)


if __name__ == "__main__":
    main()
