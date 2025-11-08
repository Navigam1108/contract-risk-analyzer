# src/clause_extractor.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import re

# --- Configuration ---
# 1. Using the official 4k-instruct model from Microsoft
MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"

# We will load the model and tokenizer once and reuse them.
model = None
tokenizer = None

def load_clause_model():
    """
    Loads the Phi-3 model in 8-bit quantization.
    """
    global model, tokenizer
    
    if model is None:
        print(f"Loading base model {MODEL_PATH} with 8-bit quantization...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            
            # 2. Load model with 8-bit quantization and device_map
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                load_in_8bit=True,    # <-- The 8-bit flag
                device_map="auto",  # <-- Automatically uses your GPU
                trust_remote_code=True,
            )
            print("Base model loaded successfully on your GPU.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure 'bitsandbytes' is installed (pip install bitsandbytes)")
            model = None # Ensure it stays None if loading failed

def extract_clauses(full_contract_text):
    """
    Extracts all legal clauses from a full contract text using the
    base Phi-3 model.
    """
    # 1. Ensure the model is loaded
    if model is None or tokenizer is None:
        load_clause_model()
        if model is None: # Check again if loading failed
            return [] 

    # 3. New Zero-Shot Prompt
    # We must be very explicit since this model is not fine-tuned.
    # We ask it to act as an expert and strictly return JSON.
    prompt = f"""
    <|system|>
    You are a text processing expert. Your task is to extract all distinct clauses from the provided contract text. A clause is a single paragraph or a distinct provision.
    You must return your answer *only* as a valid JSON list of strings. Do not provide any preamble, explanation, or conversational text. Your entire response must be the JSON list.
    Example Format: ["This is the first clause.", "This is the second clause.", "This is the third clause."]
    <|end|>
    <|user|>
    Here is the contract text:
    
    {full_contract_text}
    <|end|>
    <|assistant|>
    """

    # 3. Use a pipeline for easy generation
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096, # Adjust based on expected output size
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # The 'repetition_penalty' can help if the model repeats itself
        outputs = pipe(prompt, repetition_penalty=1.1, temperature=0.1)
        
        generated_text = outputs[0]['generated_text']
        assistant_response = generated_text.split("<|assistant|>")[-1].strip()

        # 4. Clean and parse the JSON output
        json_match = re.search(r'\[.*\]', assistant_response, re.DOTALL)
        
        if not json_match:
            print("Error: Model did not return a JSON list.")
            print("Raw response:", assistant_response)
            return []

        json_string = json_match.group(0)
        clauses = json.loads(json_string)
        
        if isinstance(clauses, list) and all(isinstance(item, str) for item in clauses):
            return clauses
        else:
            print("Error: Parsed JSON is not a list of strings.")
            return []

    except Exception as e:
        print(f"Error during clause extraction pipeline: {e}")
        return []