# app.py

import streamlit as st
import pandas as pd
from src.pdf_processor import extract_text_from_pdf
from src.clause_extractor import extract_clauses
from src.risk_analyzer import analyze_clauses_batch  # <-- IMPORT THE NEW FUNCTION

# ... (Page config and Title are the same) ...
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Contract Risk Analyzer")
st.markdown("Upload a PDF contract to automatically extract clauses, analyze risks with AI, and get plain English summaries.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    
    # --- 3. Processing Pipeline ---
    
    with st.spinner("Step 1: Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        if full_text is None:
            st.error("Could not extract text from this PDF. It might be an image-based or scanned document.")
            st.stop()
        st.success("✅ Text extracted successfully.")

    with st.spinner("Step 2: Identifying clauses..."):
        clauses = extract_clauses(full_text)
        if not clauses:
            st.error("Could not identify any clauses. The Gemini API may have had an issue.")
            st.stop()
        st.success(f"✅ Identified {len(clauses)} clauses.")

    # --- THIS IS THE NEW PART ---
    # We no longer loop. We make one single batch call.
    with st.spinner(f"Step 3: Analyzing {len(clauses)} clauses."):
        
        analysis_results = analyze_clauses_batch(clauses) # <-- ONE CALL
        
        # Now we combine the original clauses with their analyses
        results = []
        for i, clause in enumerate(clauses):
            analysis = analysis_results[i]
            results.append({
                "clause": clause,
                "risk_level": analysis.get("risk_level", "Error"),
                "risk_explanation": analysis.get("risk_explanation", "N/A"),
                "plain_english": analysis.get("plain_english", "N/A"),
                "error": analysis.get("error", None)
            })
            
        st.success("✅ Analysis complete.")
    # --- END OF NEW PART ---

    # --- 4. Display Results (This section is unchanged) ---
    st.header("Contract Analysis Dashboard")

    df = pd.DataFrame(results)

    # 4a. Risk Summary
    risk_counts = df['risk_level'].value_counts()
    cols = st.columns(4)
    with cols[0]:
        st.metric("High Risk", risk_counts.get("High", 0))
    with cols[1]:
        st.metric("Medium Risk", risk_counts.get("Medium", 0))
    with cols[2]:
        st.metric("Low Risk", risk_counts.get("Low", 0))
    with cols[3]:
        st.metric("Informational", risk_counts.get("Informational", 0))

    # 4b. Results DataGrid
    st.subheader("All Identified Clauses")
    
    def color_risk(val):
        color_map = {
            "High": "background-color: #FF4B4B; color: white;",
            "Medium": "background-color: #FFA500;",
            "Low": "background-color: #28a745; color: white;",
            "Informational": "background-color: #17a2b8; color: white;",
            "Error": "background-color: #808080; color: white;"
        }
        return color_map.get(val, "")

    display_df = df[['risk_level', 'clause', 'plain_english']].rename(columns={
        'risk_level': 'Risk',
        'clause': 'Original Clause',
        'plain_english': 'Plain English Summary'
    })
    
    st.dataframe(
        # --- FIX 2 ---
        display_df.style.map(color_risk, subset=['Risk']),
        # --- FIX 3 ---
        width='stretch',
        hide_index=True
    )

    # 4c. Detailed View (Expanders)
    st.subheader("Detailed Clause-by-Clause View")
    for index, row in df.iterrows():
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢", "Informational": "🔵", "Error": "⚫"}.get(row['risk_level'], "⚫")
        
        with st.expander(f"{icon} **{row['risk_level']} Risk:** {row['clause'][:100]}..."):
            
            st.markdown(f"**Risk Level:** `{row['risk_level']}`")
            
            st.markdown("#### Plain English Explanation")
            st.success(row['plain_english'])
            
            st.markdown("#### Risk Explanation")
            # --- FIX 1 ---
            st.info(row['risk_explanation'])
            
            st.markdown("#### Original Clause")
            st.text_area("Original", value=row['clause'], height=150, disabled=True, key=f"orig_{index}")
            
            if row['error']:
                st.error(f"An error occurred during analysis: {row['error']}")