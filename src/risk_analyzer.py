# src/risk_analyzer.py

import google.generativeai as genai
import streamlit as st
import json
import re

# We will configure the model once and reuse it.
model = None

def load_gemini_model():
    """
    Initializes and configures the Gemini model from Streamlit secrets.
    """
    global model
    try:
        # st.secrets is the modern way to access secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .streamlit/secrets.toml")
            
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.5-flash-latest')
        print("Gemini model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading Gemini model: {e}")
        # This error will be shown in the Streamlit app's console
        model = None

def analyze_clause_with_gemini(clause_text):
    """
    Analyzes a single legal clause using the Gemini API.

    Args:
        clause_text (str): The text of the legal clause.

    Returns:
        dict: A dictionary with 'risk_level', 'risk_explanation', and 'plain_english'.
              Returns a dictionary with an 'error' key on failure.
    """
    # 1. Ensure the model is loaded
    if model is None:
        load_gemini_model()
        if model is None: # Check again if loading failed
            return {"error": "Gemini model is not loaded. Check API key and configuration."}

    # 2. Define the structured prompt
    # This prompt is engineered to return a clean JSON object.
    prompt = f"""
    Analyze the following legal contract clause. Provide your analysis in a valid JSON format.

    **Analysis Rules:**
    1.  **risk_level**: Classify the risk as "Low", "Medium", "High", or "Informational".
        - "High": Poses a significant, non-standard danger or liability (e.g., unlimited liability, auto-renewal with no exit).
        - "Medium": A standard but important clause that one party should be aware of (e.g., standard confidentiality, non-compete).
        - "Low": Standard, benign, or "boilerplate" text (e.g., "This agreement is governed by the laws of Delaware").
        - "Informational": Not a risk, just a definition or statement (e.g., "The 'Company' shall mean...").
    2.  **risk_explanation**: A *brief*, 1-2 sentence explanation for the assigned risk level.
    3.  **plain_english**: A simple, 1-2 sentence rewrite of the clause in plain English.

    **Clause to Analyze:**
    ---
    {clause_text}
    ---

    **JSON Output:**
    """

    # 3. Call the API and handle response
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Low temp for more deterministic JSON output
                response_mime_type="application/json" # Ask for JSON explicitly
            )
        )
        
        # 4. Parse the JSON response
        # The model's response will be a string, which we load into a dict
        json_response = json.loads(response.text)
        
        # Basic validation of the returned JSON
        if not all(key in json_response for key in ["risk_level", "risk_explanation", "plain_english"]):
             return {"error": "Model returned incomplete JSON."}
        
        return json_response

    except Exception as e:
        print(f"Error during Gemini API call or JSON parsing: {e}")
        # This could be an API error or a json.loads error
        return {"error": f"API/Parsing error: {str(e)}"}