import streamlit as st

# ===== Must be the first Streamlit command =====
st.set_page_config(page_title="BRSR Scope Analyzer", layout="wide")

import fitz  # PyMuPDF
import re
import pandas as pd
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import plotly.express as px
from collections import Counter
import torch  # Ensure PyTorch backend

nltk.download('punkt', quiet=True)

# ====== Load Models ======
@st.cache_data(show_spinner=False)
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

@st.cache_resource
def load_sentiment_pipeline():
    # Force PyTorch to avoid TensorFlow/Keras issues
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

sentiment_pipe = load_sentiment_pipeline()

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt"
    )

summarizer = load_summarizer()

# ====== Utilities ======
def extract_text_from_pdf_filelike(filelike):
    file_bytes = filelike.read()
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    text_chunks = [page.get_text("text") for page in doc]
    return "\n\n".join(text_chunks)

def split_paragraphs(text):
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return paras if paras else [text.strip()]

def split_sentences(text):
    return sent_tokenize(text)

# ====== Keywords ======
SCOPE1_KW = [k.lower() for k in [
    "fuel", "diesel", "natural gas", "boiler", "on-site", "onsite", "vehicle", "fleet",
    "combustion", "process emissions", "fugitive", "stationary combustion", "owned vehicles",
    "on site generation", "generator", "internal combustion"
]]
SCOPE2_KW = [k.lower() for k in [
    "electricity", "purchased power", "grid", "purchased heat",
    "purchased steam", "scope 2", "electricity consumption", "renewable electricity"
]]
SCOPE3_KW = [k.lower() for k in [
    "value chain", "supply chain", "upstream", "downstream", "supplier", "vendor",
    "procurement", "raw material", "transport", "distribution", "shipping", "freight",
    "logistics", "third party", "outsourced", "contracted", "leased assets", "waste",
    "disposal", "recycling", "product use", "end-of-life", "product life cycle",
    "capital goods", "purchased goods", "fuel-and-energy-related", "investments",
    "business travel", "employee commute", "customer use", "franchise", "goods transport",
    "materials sourcing", "external partner"
]]

EMISSION_RE = re.compile(r'(\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?\s*(?:tCO2e|tonnes|tons|kg CO2e|CO2))', re.IGNORECASE)

# ====== Functions ======
def classify_scope_text(text):
    s = text.lower()

    matches_1 = [kw for kw in SCOPE1_KW if kw in s]
    matches_2 = [kw for kw in SCOPE2_KW if kw in s]
    matches_3 = [kw for kw in SCOPE3_KW if kw in s]

    if re.search(r'\bscope\s*1\b', s) or re.search(r'\bscope\s*i\b', s): 
        matches_1.append("scope 1")
    if re.search(r'\bscope\s*2\b', s) or re.search(r'\bscope\s*ii\b', s): 
        matches_2.append("scope 2")
    if (re.search(r'\bscope\s*3\b', s) or 
        re.search(r'\bscope\s*iii\b', s) or 
        re.search(r'\bcategory\s*3\b', s) or 
        re.search(r'\bcategory\s*iii\b', s) or
        re.search(r'value[-\s]*chain.*emission', s, re.IGNORECASE) or
        re.search(r'indirect\s+emissions', s, re.IGNORECASE)):
        matches_3.append("scope 3")

    counts = {"Scope 1": len(matches_1), "Scope 2": len(matches_2), "Scope 3": len(matches_3)}
    top_scope = max(counts, key=counts.get)
    total_matches = sum(counts.values())

    if total_matches == 0:
        return "Uncategorized", 0.0, []

    score = counts[top_scope] / total_matches
    if score < 0.35:
        return "Uncategorized", score, []

    return top_scope, score, list(set(matches_1 + matches_2 + matches_3))

def extract_emission_mentions(text):
    return [m.group(0).strip() for m in EMISSION_RE.finditer(text)]

def batch_sentiment(sentences, batch_size=32):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            out = sentiment_pipe(batch)
        except Exception:
            out = [{"label": "NEUTRAL", "score": 0.0} for _ in batch]
        results.extend(out)

    unified = []
    for sent, r in zip(sentences, results):
        label = r.get("label", "NEUTRAL")
        score = float(r.get("score", 0.0))
        polarity = 1.0 if "POS" in label.upper() else -1.0 if "NEG" in label.upper() else 0.0
        unified.append({"sentence": sent, "label": label, "score": score, "polarity": polarity})
    return unified

def extract_entities(text, top_k=30):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    c = Counter(ents)
    rows = [{"entity": e, "label": l, "count": c[(e, l)]} for e, l in c]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k)
    return df

def summarize_text(text, word_count=120):
    try:
        max_len = min(512, word_count)
        summary = summarizer(text, max_length=max_len, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        sents = split_sentences(text)
        return " ".join(sents[:min(4, len(sents))])

# ====== Streamlit UI ======
st.title("BRSR Scope Analyzer 🌱")

with st.sidebar:
    st.header("Options")
    show_examples = st.checkbox("Show example keywords", value=False)
    word_count = st.slider("Summary word count", 50, 400, 150, 25)
    show_uncategorized = st.checkbox("Show Uncategorized sentences", value=True)

if show_examples:
    st.markdown("**Scope 1:** " + ", ".join(SCOPE1_KW[:10]))
    st.markdown("**Scope 2:** " + ", ".join(SCOPE2_KW[:10]))
    st.markdown("**Scope 3:** " + ", ".join(SCOPE3_KW[:10]))

uploaded = st.file_uploader("Upload BRSR PDF", type=["pdf"])
if uploaded:
    st.info("Extracting text...")
    text = extract_text_from_pdf_filelike(uploaded)
    if not text.strip():
        st.error("No text extracted. Try another file.")
        st.stop()

    st.success("Text extracted successfully!")
    st.write(summarize_text(text, word_count))

    paras = split_paragraphs(text)
    sentences = [s for p in paras for s in split_sentences(p)]

    rows = []
    for sent in sentences:
        scope, score, matched = classify_scope_text(sent)
        emissions = extract_emission_mentions(sent)
        rows.append({
            "sentence": sent,
            "scope": scope,
            "matches": "; ".join(matched),
            "emissions": "; ".join(emissions),
            "scope_score": score
        })
    df = pd.DataFrame(rows)

    if not show_uncategorized:
        df = df[df["scope"] != "Uncategorized"]

    sentiments = batch_sentiment(df["sentence"].tolist())
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(sentiments)], axis=1)

    st.subheader("Scope Distribution")
    counts = df["scope"].value_counts().reindex(["Scope 1", "Scope 2", "Scope 3", "Uncategorized"]).fillna(0)
    fig = px.pie(values=counts.values, names=counts.index, title="Scope Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Named Entities (top)")
    st.dataframe(extract_entities(text, top_k=40))

    st.subheader("Inspect sentences by scope")
    for scope in ["Scope 1", "Scope 2", "Scope 3", "Uncategorized"]:
        subset = df[df["scope"] == scope]
        if subset.empty:
            continue
        with st.expander(f"{scope} ({len(subset)} sentences)"):
            display_cols = [c for c in ["sentence", "scope_score", "label", "score", "polarity", "emissions"] if c in subset.columns]
            display_df = subset[display_cols].copy()
            display_df.rename(columns={"score": "sentiment_score"}, inplace=True)
            display_df = display_df.loc[:, ~display_df.columns.duplicated()].copy()
            st.dataframe(display_df.head(200))

    st.download_button(
        "Download full CSV",
        df.to_csv(index=False),
        file_name="brsr_scope_analysis.csv",
        mime="text/csv"
    )

    st.success("✅ Analysis completed successfully.")
else:
    st.info("Upload a BRSR PDF on the left to begin.")
