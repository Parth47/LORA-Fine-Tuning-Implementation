Simplified LoRA Implementation for BLOOM-3B

This repository contains a simplified implementation for fine-tuning the bigscience/bloom-3b model using Low-Rank Adaptation (LoRA). 
The process is demonstrated by fine-tuning the model on the SQuAD v2 dataset for question-answering tasks.
The original implementation was developed in a Jupyter Notebook. 
This repository refactors that code into a more structured and reusable format.
Features8-bit Quantization: Utilizes bitsandbytes for efficient 8-bit model quantization, reducing memory footprint.
PEFT (Parameter-Efficient Fine-Tuning): Leverages the peft library from Hugging Face to apply LoRA.
Clear Structure: The code is organized into distinct modules for model setup, training, and inference.Repository Structure.


├── .gitignore         # Specifies files to be ignored by Git
├── README.md          # This file
├── requirements.txt   # Project dependencies
└── src
    ├── model_setup.py # Handles base model loading and quantization
    ├── train.py       # Main script for training the LoRA model
    └── inference.py   # Script for running inference with the trained model
    
How to Use
1. Installation
First, clone the repository and navigate into the directory:git clone <your-repo-url>
cd <your-repo-name>
Next, install the required dependencies:pip install -r requirements.txt

2. Training
 To start the fine-tuning process, run the train.py script.
This script will:Load the quantized base model (bigscience/bloom-3b).
Prepare the SQuAD v2 dataset.Apply the LoRA configuration.Run the training using the transformers.
Trainer.python src/train.py
The trained LoRA adapter weights will be saved to the outputs directory.
You can optionally push the final model to the Hugging Face Hub by modifying the script.

3. Inference
After training, you can use the inference.py script to test your fine-tuned model.
This script loads the base model and applies the saved LoRA adapter for inference.
Make sure to update the peft_model_path in src/inference.py to point to your trained adapter (e.g., outputs/checkpoint-100).python src/inference.py
You can modify the context and question variables within the script to ask your own questions.
