# Clinical NLP Proof of Concept

**Cotiviti Intern Assessment — Topic 1: Clinical Natural Language Technology for Health Care**

Video Presentation: https://youtu.be/D2ymV8k3gW0

A Streamlit-based demonstration of Clinical NLP applied to healthcare documentation, showcasing three core capabilities:

1. **Medical Named Entity Recognition (NER)** — Extracts conditions, medications, procedures, anatomical findings, and lab results from free-text clinical notes.
2. **Clinical Note Summarization** — Generates structured summaries suitable for clinical handoffs and chart review.
3. **ICD-10 Code Suggestion** — Proposes relevant diagnostic codes with confidence levels to assist medical coders.

## Architecture

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Streamlit UI │───▶│  NLP Pipeline │───▶│  LLM Provider    │
│  (Input/View) │◀───│  (Prompting)  │◀───│  (Claude/GPT/    │
│               │    │               │    │   Ollama)         │
└──────────────┘    └───────────────┘    └──────────────────┘
                           │
                    ┌──────┴──────┐
                    │  JSON Parse │
                    │  & Render   │
                    └─────────────┘
```

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd poc_demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Enter your API key in the sidebar
```

### Recommended: Groq (Free, Fast)

1. Sign up at [console.groq.com](https://console.groq.com) (instant, free)
2. Create an API key
3. Select **"Groq (Free)"** in the app sidebar and paste your key
4. Uses `llama-3.3-70b-versatile` — fast inference, strong JSON output

### Alternative: Ollama (Local, No API Key Needed)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:3b
# Select "Ollama (Local)" in the app sidebar
```

## Features

- **Three sample clinical notes** included (ED visit, primary care follow-up, surgical discharge)
- **Custom text input** — paste any clinical note
- **Multi-provider support** — Groq (free, recommended), Anthropic Claude, OpenAI GPT, or local Ollama
- **Structured output** — Entities displayed as categorized cards, ICD codes with confidence indicators
- **Performance timing** — Execution time displayed for each NLP task

## Relevance to Cotiviti

This POC demonstrates technologies directly applicable to Cotiviti's core business:

- **Payment Integrity** — Automated entity extraction from clinical records supports DRG validation and claim review
- **Medical Record Coding** — ICD-10 suggestion capability augments human coders in Clinical Chart Validation
- **Fraud Detection** — NER can flag inconsistencies between documented conditions and billed codes
- **Operational Efficiency** — Clinical summarization reduces time spent on chart review

## Tech Stack

- Python 3.10+
- Streamlit (UI framework)
- Groq / Anthropic / OpenAI / Ollama (LLM providers)
- JSON structured output parsing

## Author

Gaurav Gurunath Choughule  
MS Artificial Intelligence, Northeastern University (Khoury College of Computer Sciences)
