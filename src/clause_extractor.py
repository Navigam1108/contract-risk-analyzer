# src/clause_extractor.py

import google.generativeai as genai
import streamlit as st
import json
import re

# We will configure a model for this task
model = None

def load_extraction_model():
    """
    Initializes and configures the Gemini model from Streamlit secrets.
    """
    global model
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets.")
            
        genai.configure(api_key=api_key)
        
        # Using gemini-2.5-flash as requested
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini 2.5 Flash (for extraction) loaded successfully.")
        
    except Exception as e:
        print(f"Error loading Gemini model: {e}")
        model = None

def extract_clauses(full_contract_text):
    """
    Extracts all legal clauses from a full contract text using the
    Gemini API.
    """
    if model is None:
        load_extraction_model()
        if model is None:
            return [] 

    # 2. Create the extraction prompt
    prompt = f"""
    You are an expert legal assistant. Your task is to extract all distinct clauses from the provided contract text.

    **Rules:**
    1.  A "clause" is a full paragraph of text, often starting with a number or letter (e.g., "1.1", "a.").
    2.  You MUST ignore titles, headers, footers, page numbers, and source citations (e.g., "Source: LINK PLUS CORP").
    3.  You MUST ignore simple lists of names or definitions that are not full clauses.
    4.  You MUST return your answer *only* as a valid JSON list of strings.
    5.  Do not provide any preamble, explanation, or conversational text. Your entire response must be the JSON list.

    Example Format: ["This is the first clause.", "This is the second clause.", "This is the third clause."]

    **Contract Text to Analyze:**
    ---
    {full_contract_text}
    ---

    **JSON Output:**
    """

    # 3. Call the API
    try:
        print("Sending full contract to Gemini 2.5 Flash for clause extraction...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # 4. Parse the JSON response
        json_response = json.loads(response.text)
        
        if isinstance(json_response, list) and all(isinstance(item, str) for item in json_response):
            print(f"Successfully extracted {len(json_response)} clauses.")
            return json_response
        else:
            print("Error: Parsed JSON is not a list of strings.")
            return []

    except Exception as e:
        print(f"Error during Gemini API call for extraction: {e}")
        return []