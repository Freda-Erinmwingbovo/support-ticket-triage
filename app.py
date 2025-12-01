# ============================================================
# app.py — FIXED & STREAMLIT-CLOUD SAFE — DEC 01 2025
# ============================================================
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download
import os
import re

st.set_page_config(
    page_title="ML Support Brain",
    page_icon="🚀",
    layout="wide",
)

# -------------------------------------------------------
# MODEL LOADING (STREAMLIT CLOUD SAFE)
# -------------------------------------------------------
@st.cache_resource
def load_models():
    repo = "FredaErins/support-triage-models"

    try:
        p_type = hf_hub_download(repo_id=repo, filename="ticket_type_classifier_PROD_compressed.pkl")
        p_priority = hf_hub_download(repo_id=repo, filename="priority_classifier_PROD_compressed.pkl")
        p_queue = hf_hub_download(repo_id=repo, filename="queue_routing_PROD_compressed.pkl")  # FIXED NAME
    except Exception as e:
        st.error(f"❌ Failed to download models: {e}")
        st.stop()

    return (
        joblib.load(p_type),
        joblib.load(p_priority),
        joblib.load(p_queue)
    )


model_type, model_priority, model_queue = load_models()

# -------------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------------
def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop = {
        "a","an","the","and","or","is","are","was","were","in","on","at","to",
        "for","with","of","this","that","these","those","i","you","he","she",
        "it","we","they","my","your","his","her","its","our","their","from",
        "as","by","be","been","am","will","can","do","does","did","have",
        "has","had","not","but","if","then","so","no","yes"
    }
    return " ".join(w for w in t.split() if w not in stop)

# -------------------------------------------------------
# PREDICT
# -------------------------------------------------------
def predict_ticket(subject, body, queue_hint, th_priority=0.8, th_queue=0.85):

    text = clean_text(subject + " " + body)

    df0 = pd.DataFrame([{
        "text": text,
        "queue": queue_hint if queue_hint else "General",
        "priority": "Medium"
    }])

    # --- TYPE ---
    type_pred = model_type.predict(df0)[0]
    type_conf = model_type.predict_proba(df0)[0].max()

    # --- PRIORITY ---
    df0["ticket_type"] = type_pred
    df_p = df0[["text", "queue", "ticket_type"]]
    pr_pred = model_priority.predict(df_p)[0]
    pr_conf = model_priority.predict_proba(df_p)[0].max()

    # --- QUEUE ---
    df_q = pd.DataFrame([{
        "text": text,
        "ticket_type": type_pred,
        "priority": pr_pred
    }])
    q_pred = model_queue.predict(df_q)[0]
    q_conf = model_queue.predict_proba(df_q)[0].max()

    # --- AUTO LOGIC ---
    auto_p = pr_pred if pr_conf >= th_priority else None
    auto_q = q_pred if q_conf >= th_queue else None

    # --- FINAL ACTION ---
    if auto_p and auto_q:
        action = "FULLY AUTO-TRIAGED → No human needed"
    elif auto_q:
        action = "AUTO-ROUTED → Agent only confirms priority"
    elif auto_p:
        action = "AUTO-PRIORITY → Agent confirms queue"
    elif type_conf >= 0.90:
        action = "AUTO-TYPE ONLY → Agent decides priority & queue"
    else:
        action = "HUMAN REVIEW SUGGESTED → Low confidence"

    return {
        "ticket_type": type_pred,
        "type_confidence": float(type_conf),
        "predicted_priority": pr_pred,
        "priority_confidence": float(pr_conf),
        "auto_set_priority": auto_p,
        "predicted_queue": q_pred,
        "queue_confidence": float(q_conf),
        "auto_route_to": auto_q,
        "final_action": action
    }

# -------------------------------------------------------
# SAFE LOGGING (no file system issues)
# -------------------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []


def log_prediction(subject, result):
    st.session_state.logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject[:100],
        "ticket_type": result["ticket_type"],
        "priority": result["auto_set_priority"] or result["predicted_priority"],
        "queue": result["auto_route_to"] or result["predicted_queue"],
        "auto_queue": bool(result["auto_route_to"]),
        "action": result["final_action"],
    })


# -------------------------------------------------------
# UI
# -------------------------------------------------------
with st.sidebar:
    st.title("🚀 ML Support Brain")
    st.caption("Type → Priority → Queue • 88.8% accuracy")

    total = len(st.session_state.logs)
    st.metric("Tickets Processed", total)
    if total > 0:
        routed_rate = sum(1 for x in st.session_state.logs if x["auto_queue"]) / total * 100
        st.metric("Auto-Routed Rate", f"{routed_rate:.1f}%")

    st.divider()
    th_p = st.slider("Auto-Priority Threshold", 0.50, 1.00, 0.80, 0.01)
    th_q = st.slider("Auto-Queue Threshold", 0.50, 1.00, 0.85, 0.01)

st.title("Live Support Ticket Auto-Triage Engine")
st.write("Enter a customer message to classify its type, priority, and routing.")

col1, col2 = st.columns([2, 1])
with col1:
    subject = st.text_input("Subject", "")
    body = st.text_area("Body", "", height=160)

with col2:
    queue_hint = st.text_input("Current Queue (optional)", "")

if st.button("TRIAGE THIS TICKET", use_container_width=True):
    if not subject.strip() or not body.strip():
        st.warning("Subject and Body are required.")
        st.stop()

    with st.spinner("Analysing…"):
        result = predict_ticket(subject, body, queue_hint, th_p, th_q)
        log_prediction(subject, result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Type", result["ticket_type"], f"{result['type_confidence']:.1%}")
    c2.metric("Priority",
              (result["auto_set_priority"] or result["predicted_priority"]).upper(),
              f"{result['priority_confidence']:.1%}")
    c3.metric("Queue",
              result["auto_route_to"] or result["predicted_queue"],
              f"{result['queue_confidence']:.1%}")

    st.subheader(result["final_action"])
    if result["auto_route_to"]:
        st.balloons()

st.caption("Built solo in 3.5 weeks • Production-ready.")
