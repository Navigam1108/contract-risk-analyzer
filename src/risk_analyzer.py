# src/risk_analyzer.py

import google.generativeai as genai
import streamlit as st
import json
import re

model = None

def load_gemini_model():
    """
    Initializes and configures the Gemini model from Streamlit secrets.
    """
    global model
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets.")
            
        genai.configure(api_key=api_key)
        
        # Use gemini-2.5-flash as requested
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini 2.5 Flash (for analysis) loaded successfully.")
        
    except Exception as e:
        print(f"Error loading Gemini model: {e}")
        model = None

def analyze_clauses_batch(clauses_list):
    """
    Analyzes a BATCH of legal clauses in a single API call.

    Args:
        clauses_list (list): A list of clause strings.

    Returns:
        list: A list of dictionaries, one for each clause's analysis.
              Returns a list of error dictionaries on failure.
    """
    if model is None:
        load_gemini_model()
        if model is None:
            return [{"error": "Gemini model not loaded"} for _ in clauses_list]

    # Convert the list of clauses into a numbered string
    formatted_clauses = ""
    for i, clause in enumerate(clauses_list):
        formatted_clauses += f"CLAUSE {i+1}:\n\"{clause}\"\n\n"

    # 2. Define the structured BATCH prompt
    prompt = f"""
    You are an expert legal analyst. Analyze every clause provided below.
    Return a valid JSON list, where each item in the list is a JSON object
    containing the analysis for the corresponding clause.

    **Analysis Rules (for each clause):**
    1.  **risk_level**: Classify as "Low", "Medium", "High", or "Informational".
    2.  **risk_explanation**: A *brief*, 1-2 sentence explanation.
    3.  **plain_english**: A simple, 1-2 sentence rewrite.

    **Example Output Format:**
    [
      {{"risk_level": "Low", "risk_explanation": "...", "plain_english": "..."}},
      {{"risk_level": "High", "risk_explanation": "...", "plain_english": "..."}}
    ]

    **Clauses to Analyze:**
    ---
    {formatted_clauses}
    ---

    **JSON Output (must be a list of {len(clauses_list)} objects):**
    """

    # 3. Call the API (once)
    try:
        print(f"Sending batch of {len(clauses_list)} clauses to Gemini 2.5 Flash...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        
        # 4. Parse the JSON list response
        json_response = json.loads(response.text)
        
        if isinstance(json_response, list) and len(json_response) == len(clauses_list):
            return json_response
        else:
            print("Error: Model did not return a valid list or list length mismatch.")
            return [{"error": "Batch analysis failed"} for _ in clauses_list]

    except Exception as e:
        print(f"Error during Gemini API batch call: {e}")
        return [{"error": f"API/Parsing error: {str(e)}"} for _ in clauses_list]