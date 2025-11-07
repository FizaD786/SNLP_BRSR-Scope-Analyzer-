# BRSR Scope Analyzer 🌱

**BRSR Scope Analyzer** is a Streamlit web application that automatically extracts, classifies, and analyzes sustainability disclosures from **BRSR (Business Responsibility and Sustainability Report) PDFs**. It focuses on Scope 1, 2, and 3 emissions, performs sentiment analysis, summarizes content, detects named entities, and visualizes insights interactively.

---

## Features

- **PDF Text Extraction:** Extracts text from uploaded BRSR PDFs using PyMuPDF.  
- **Scope Classification:** Categorizes sentences into **Scope 1, Scope 2, Scope 3**, or Uncategorized.  
- **Emission Extraction:** Detects emission values like tCO2e, kg CO2e, tonnes, etc.  
- **Sentiment Analysis:** Performs sentence-level sentiment analysis using Hugging Face Transformers.  
- **Summarization:** Generates concise summaries of reports.  
- **Named Entity Recognition:** Extracts top entities for deeper insights using spaCy.  
- **Visualization:** Interactive plots of scope distribution with Plotly.  
- **CSV Download:** Export analyzed data for further use.  
- **PyTorch Backend:** Ensures compatibility and performance for transformer models.  

---

## Tech Stack

- **Python 3.11**  
- **[Streamlit](https://streamlit.io/)** – Web interface  
- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** – PDF extraction  
- **[spaCy](https://spacy.io/)** – NLP & entity recognition  
- **[Hugging Face Transformers](https://huggingface.co/transformers/)** – Sentiment analysis & summarization  
- **[NLTK](https://www.nltk.org/)** – Sentence tokenization  
- **[Plotly](https://plotly.com/python/)** – Interactive charts  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/brsr-scope-analyzer.git
cd brsr-scope-analyzer
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

