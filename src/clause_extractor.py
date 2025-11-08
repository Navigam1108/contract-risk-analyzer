# src/clause_extractor.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import re

# --- Configuration ---
# TODO: Update this path to your fine-tuned model
# It could be a local path: "./models/my-phi-3-finetuned"
# Or a Hugging Face hub name: "my-username/my-phi-3-finetuned"
MODEL_PATH = "your-username/phi-3.5-mini-instruct-finetuned" 

# We will load the model and tokenizer once and reuse them.
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clause_model():
    """
    Loads the fine-tuned model and tokenizer into memory.
    """
    global model, tokenizer
    
    if model is None:
        print(f"Loading clause extraction model from {MODEL_PATH} to {device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
                device_map=device,
                trust_remote_code=True,
            )
            print("Clause model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check that MODEL_PATH in src/clause_extractor.py is correct.")
            model = None # Ensure it stays None if loading failed

def extract_clauses(full_contract_text):
    """
    Extracts all legal clauses from a full contract text using the
    fine-tuned Phi-3.5 model.

    Args:
        full_contract_text (str): The complete text from the PDF.

    Returns:
        list: A list of strings, where each string is an extracted clause.
              Returns an empty list on failure.
    """
    # 1. Ensure the model is loaded
    if model is None or tokenizer is None:
        load_clause_model()
        if model is None: # Check again if loading failed
            return [] 

    # 2. Create the instruction prompt
    # This prompt asks the model to return a JSON list of strings,
    # which is the easiest format to parse.
    prompt = f"""
    <|system|>
    You are an expert legal assistant. Your task is to extract all individual clauses from the contract text provided.
    A clause is a distinct section, paragraph, or provision.
    Return your answer *only* as a valid JSON list of strings.
    Example: ["Clause 1 text...", "Clause 2 text...", "Clause 3 text..."]
    <|end|>
    <|user|>
    Extract all clauses from the following contract:

    {full_contract_text}
    <|end|>
    <|assistant|>
    """

    # 3. Use a pipeline for easy generation
    # Using a pipeline handles tokenization, generation, and decoding
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096, # Adjust based on expected output size
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # The 'repetition_penalty' can help if the model repeats itself
        outputs = pipe(prompt, repetition_penalty=1.1)
        
        # The pipeline returns a list, we take the first result
        generated_text = outputs[0]['generated_text']
        
        # The model's output will include the prompt, so we find the assistant's part
        assistant_response = generated_text.split("<|assistant|>")[-1].strip()

        # 4. Clean and parse the JSON output
        # Models sometimes add markdown ```json ... ``` or other text.
        json_match = re.search(r'\[.*\]', assistant_response, re.DOTALL)
        
        if not json_match:
            print("Error: Model did not return a JSON list.")
            print("Raw response:", assistant_response)
            return []

        json_string = json_match.group(0)
        clauses = json.loads(json_string)
        
        # Final check to ensure it's a list of strings
        if isinstance(clauses, list) and all(isinstance(item, str) for item in clauses):
            return clauses
        else:
            print("Error: Parsed JSON is not a list of strings.")
            return []

    except Exception as e:
        print(f"Error during clause extraction pipeline: {e}")
        return []