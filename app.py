"""
Clinical NLP Proof of Concept
=============================
A Streamlit-based demonstration of Clinical Natural Language Processing applied 
to healthcare documentation. This prototype showcases medical entity extraction 
(NER), clinical note summarization, and ICD-10 code suggestion using LLMs.

Author: Gaurav Gurunath Choughule
Cotiviti Intern Assessment — Topic 1: Clinical NLP for Health Care

Usage:
    pip install streamlit anthropic
    streamlit run app.py
"""

import streamlit as st
import json
import re
import time
import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Configuration ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical NLP Demo — Cotiviti POC",
    page_icon="🏥",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .entity-card {
        background-color: #f8f9fa;
        border-left: 4px solid;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        color: #1a1a2e !important;
    }
    .entity-card strong {
        color: #1a1a2e !important;
    }
    .entity-card small {
        color: #4a4a5a !important;
    }
    .entity-condition { border-color: #dc3545; }
    .entity-medication { border-color: #0d6efd; }
    .entity-procedure { border-color: #198754; }
    .entity-anatomy { border-color: #6f42c1; }
    .entity-lab { border-color: #fd7e14; }
    .icd-code {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #b8daff;
        padding: 4px 10px;
        border-radius: 16px;
        margin: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #1a1a2e !important;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-box h2 { color: white; margin: 0; font-size: 2rem; }
    .metric-box p { color: #e0e0e0; margin: 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Sample Clinical Notes ──────────────────────────────────────────────────

SAMPLE_NOTES = {
    "Emergency Department Visit": """CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: 
A 67-year-old male presents to the ED with acute onset substernal chest pain radiating to the left arm, associated with diaphoresis and dyspnea. Symptoms began approximately 2 hours ago while at rest. Patient has a history of hypertension, type 2 diabetes mellitus, and hyperlipidemia. Current medications include metformin 1000mg BID, lisinopril 20mg daily, and atorvastatin 40mg daily.

PHYSICAL EXAMINATION:
BP 158/92 mmHg, HR 98 bpm, RR 22, SpO2 94% on room air. Patient appears diaphoretic and in moderate distress. Cardiac exam reveals S1/S2 with no murmurs. Lungs with bilateral basilar crackles.

ASSESSMENT AND PLAN:
1. Acute coronary syndrome — obtain STAT troponin, 12-lead ECG, chest X-ray. Administer aspirin 325mg, nitroglycerin SL, and heparin drip per ACS protocol.
2. Acute decompensated heart failure — IV furosemide 40mg, monitor I/O.
3. Hypertensive urgency — continue home antihypertensives, titrate as needed.
4. Consult cardiology for possible cardiac catheterization.""",

    "Primary Care Follow-Up": """SUBJECTIVE:
45-year-old female presents for routine follow-up of type 2 diabetes and hypothyroidism. Reports good compliance with medications. Denies polyuria, polydipsia, or weight changes. Occasional fatigue attributed to work stress. Last HbA1c was 7.2% three months ago.

MEDICATIONS: Metformin 500mg BID, Levothyroxine 75mcg daily, Vitamin D3 2000 IU daily.

OBJECTIVE:
Vitals: BP 128/78, HR 72, BMI 28.4. Thyroid non-tender, no nodules. Monofilament testing intact bilateral feet. Fundoscopic exam normal.

LABS: HbA1c 6.8% (improved), TSH 2.4 mIU/L (normal), Lipid panel: TC 195, LDL 118, HDL 52, TG 125. Creatinine 0.9, eGFR >60.

ASSESSMENT/PLAN:
1. Type 2 diabetes — well controlled, continue metformin. Repeat HbA1c in 3 months. Reinforce diet and exercise.
2. Hypothyroidism — stable on current levothyroxine dose. Recheck TSH in 6 months.
3. Hyperlipidemia — borderline LDL. Initiate lifestyle modifications. Consider statin if LDL remains >130 at next visit.
4. Preventive care — schedule mammogram, update flu vaccine.""",

    "Surgical Discharge Summary": """DISCHARGE SUMMARY

PATIENT: 72-year-old female
ADMISSION DATE: 06/15/2026
DISCHARGE DATE: 06/20/2026

PRINCIPAL DIAGNOSIS: Right total knee arthroplasty for severe osteoarthritis.

HOSPITAL COURSE:
Patient underwent elective right total knee replacement under spinal anesthesia without complications. Post-operative course was notable for adequate pain control with femoral nerve block and PCA morphine, transitioned to oral oxycodone-acetaminophen 5/325mg Q6H PRN. Physical therapy initiated POD#1 with ambulation using a walker. Achieved 90 degrees of flexion by POD#4. DVT prophylaxis with enoxaparin 40mg SQ daily. Surgical wound clean, dry, intact with staples.

DISCHARGE MEDICATIONS:
1. Oxycodone-acetaminophen 5/325mg Q6H PRN pain
2. Enoxaparin 40mg SQ daily x 14 days
3. Aspirin 81mg daily
4. Continue home medications: amlodipine 5mg, omeprazole 20mg

FOLLOW-UP: Orthopedics in 2 weeks for staple removal. Continue home PT 3x/week."""
}

# ─── LLM Integration ────────────────────────────────────────────────────────

def call_llm(prompt, system_prompt, api_key, model_provider):
    """Call the selected LLM provider and return the response text."""
    
    if model_provider == "Anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif model_provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif model_provider == "Groq (Free)":
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif model_provider == "Ollama (Local)":
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            return response.json().get("response", "No response")
        except Exception as e:
            return f"Error: {str(e)}"

# ─── NLP Task Prompts ───────────────────────────────────────────────────────

NER_SYSTEM = """You are a clinical NLP system that extracts medical named entities from clinical notes.
Extract entities into these categories: CONDITIONS, MEDICATIONS, PROCEDURES, ANATOMY, LAB_RESULTS.
Return ONLY valid JSON in this exact format (no markdown, no backticks):
{
  "CONDITIONS": [{"entity": "...", "context": "brief context from note"}],
  "MEDICATIONS": [{"entity": "...", "dosage": "if mentioned", "frequency": "if mentioned"}],
  "PROCEDURES": [{"entity": "...", "status": "planned/completed/recommended"}],
  "ANATOMY": [{"entity": "...", "finding": "associated finding"}],
  "LAB_RESULTS": [{"entity": "...", "value": "if available", "interpretation": "normal/abnormal/critical"}]
}"""

SUMMARY_SYSTEM = """You are a clinical documentation specialist. Summarize the clinical note into a 
structured format with these sections: PATIENT OVERVIEW (1-2 sentences), KEY FINDINGS (bullet points), 
ACTIVE PROBLEMS (numbered list), and PLAN HIGHLIGHTS (bullet points). Be concise and clinically precise.
Use plain text with clear section headers."""

ICD_SYSTEM = """You are a medical coding assistant. Based on the clinical note, suggest relevant ICD-10-CM 
codes with their descriptions. Return ONLY valid JSON (no markdown, no backticks):
{
  "codes": [
    {"code": "I21.0", "description": "ST elevation myocardial infarction involving left main coronary artery", "confidence": "high/medium/low"},
    ...
  ]
}
Suggest 3-8 codes ranked by relevance. Include both primary and secondary diagnoses."""

# ─── Utility Functions ──────────────────────────────────────────────────────

def parse_json_response(text):
    """Robustly parse JSON from LLM responses, handling markdown fences."""
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None

def render_entities(entities_data):
    """Render extracted entities with styled cards."""
    if not entities_data:
        st.error("Could not parse entity extraction results. The LLM response was not valid JSON.")
        return
    
    category_config = {
        "CONDITIONS": ("🔴", "condition", "Conditions & Diagnoses"),
        "MEDICATIONS": ("💊", "medication", "Medications"),
        "PROCEDURES": ("🔧", "procedure", "Procedures"),
        "ANATOMY": ("🫁", "anatomy", "Anatomical Findings"),
        "LAB_RESULTS": ("🔬", "lab", "Lab Results"),
    }
    
    total_entities = sum(len(v) for v in entities_data.values() if isinstance(v, list))
    
    cols = st.columns(5)
    for i, (cat, (icon, css_class, label)) in enumerate(category_config.items()):
        count = len(entities_data.get(cat, []))
        cols[i].metric(f"{icon} {label}", count)
    
    st.markdown(f"**Total entities extracted: {total_entities}**")
    st.divider()
    
    for cat, (icon, css_class, label) in category_config.items():
        items = entities_data.get(cat, [])
        if items:
            st.subheader(f"{icon} {label}")
            for item in items:
                entity_name = item.get("entity", "Unknown")
                details = {k: v for k, v in item.items() if k != "entity" and v}
                detail_str = " | ".join(f"**{k.title()}:** {v}" for k, v in details.items())
                st.markdown(
                    f'<div class="entity-card entity-{css_class}">'
                    f'<strong>{entity_name}</strong><br/>'
                    f'<small>{detail_str}</small></div>',
                    unsafe_allow_html=True
                )

def render_icd_codes(icd_data):
    """Render ICD-10 code suggestions."""
    if not icd_data or "codes" not in icd_data:
        st.error("Could not parse ICD-10 suggestions.")
        return
    
    confidence_colors = {"high": "#28a745", "medium": "#ffc107", "low": "#dc3545"}
    
    for item in icd_data["codes"]:
        code = item.get("code", "???")
        desc = item.get("description", "")
        conf = item.get("confidence", "medium").lower()
        color = confidence_colors.get(conf, "#6c757d")
        
        col1, col2, col3 = st.columns([1, 4, 1])
        col1.markdown(f'<span class="icd-code"><strong>{code}</strong></span>', unsafe_allow_html=True)
        col2.write(desc)
        col3.markdown(f'<span style="color:{color}; font-weight:600;">●  {conf.title()}</span>', unsafe_allow_html=True)

# ─── Main Application ───────────────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🏥 Clinical NLP Proof of Concept</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Demonstrating Medical Entity Extraction, Clinical Summarization, '
        'and ICD-10 Code Suggestion using Large Language Models<br/>'
        '<em>Cotiviti Intern Assessment — Gaurav Gurunath Choughule</em></p>',
        unsafe_allow_html=True,
    )
    
    # ── Sidebar: Configuration ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_provider = st.selectbox(
            "LLM Provider",
            ["Groq (Free)", "Anthropic", "OpenAI", "Ollama (Local)"],
            help="Groq offers a free API key at console.groq.com. Ollama runs locally without any key."
        )
        
        api_key = ""
        if model_provider != "Ollama (Local)":
            env_key_map = {
                "Groq (Free)": "GROQ_API_KEY",
                "Anthropic": "ANTHROPIC_API_KEY",
                "OpenAI": "OPENAI_API_KEY",
            }
            env_var = env_key_map.get(model_provider, "")
            default_key = os.environ.get(env_var, "")
            api_key = st.text_input(
                f"{model_provider} API Key",
                value=default_key,
                type="password",
                help=f"Auto-loaded from .env ({env_var}) if available.",
            )
        
        st.divider()
        st.markdown("### 📋 About This Demo")
        st.markdown(
            "This prototype demonstrates three core clinical NLP capabilities:\n\n"
            "1. **Medical NER** — Extracts conditions, medications, procedures, anatomy, "
            "and lab results from free-text clinical notes.\n\n"
            "2. **Clinical Summarization** — Generates structured summaries "
            "suitable for handoffs and chart review.\n\n"
            "3. **ICD-10 Suggestion** — Proposes relevant diagnostic codes "
            "with confidence levels to assist medical coders."
        )
        st.divider()
        st.markdown(
            "**Tech Stack:** Python, Streamlit, LLM APIs\n\n"
            "**Relevant to Cotiviti:** Payment integrity, DRG validation, "
            "Clinical Chart Validation, medical record coding."
        )
    
    # ── Input Section ──
    st.subheader("📝 Clinical Note Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Use a sample note", "Enter custom text"],
        horizontal=True,
    )
    
    if input_method == "Use a sample note":
        selected_sample = st.selectbox("Select a sample note:", list(SAMPLE_NOTES.keys()))
        clinical_note = SAMPLE_NOTES[selected_sample]
    else:
        clinical_note = ""
    
    clinical_note = st.text_area(
        "Clinical Note",
        value=clinical_note,
        height=250,
        placeholder="Paste or type a clinical note here...",
    )
    
    if not clinical_note.strip():
        st.info("👆 Enter or select a clinical note above to get started.")
        return
    
    # ── Check API key ──
    if model_provider != "Ollama (Local)" and not api_key:
        st.warning(f"⚠️ Please enter your {model_provider} API key in the sidebar to proceed.")
        return
    
    # ── Analysis Buttons ──
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    run_ner = col1.button("🔍 Extract Entities", use_container_width=True)
    run_summary = col2.button("📋 Summarize Note", use_container_width=True)
    run_icd = col3.button("🏷️ Suggest ICD-10 Codes", use_container_width=True)
    run_all = col4.button("🚀 Run All", type="primary", use_container_width=True)
    
    if run_all:
        run_ner = run_summary = run_icd = True
    
    # ── Results Section ──
    if run_ner:
        st.subheader("🔍 Medical Named Entity Recognition")
        with st.spinner("Extracting medical entities..."):
            start = time.time()
            result = call_llm(clinical_note, NER_SYSTEM, api_key, model_provider)
            elapsed = time.time() - start
        
        st.caption(f"⏱️ Completed in {elapsed:.1f}s")
        entities = parse_json_response(result)
        if entities:
            render_entities(entities)
        else:
            st.error("Failed to parse entities. Raw response:")
            st.code(result)
    
    if run_summary:
        st.subheader("📋 Clinical Note Summary")
        with st.spinner("Generating structured summary..."):
            start = time.time()
            result = call_llm(clinical_note, SUMMARY_SYSTEM, api_key, model_provider)
            elapsed = time.time() - start
        
        st.caption(f"⏱️ Completed in {elapsed:.1f}s")
        st.markdown(result)
    
    if run_icd:
        st.subheader("🏷️ ICD-10-CM Code Suggestions")
        with st.spinner("Suggesting diagnostic codes..."):
            start = time.time()
            result = call_llm(clinical_note, ICD_SYSTEM, api_key, model_provider)
            elapsed = time.time() - start
        
        st.caption(f"⏱️ Completed in {elapsed:.1f}s")
        icd_data = parse_json_response(result)
        if icd_data:
            render_icd_codes(icd_data)
        else:
            st.error("Failed to parse ICD-10 suggestions. Raw response:")
            st.code(result)

if __name__ == "__main__":
    main()