# app.py — FIXED FOR STREAMLIT DEPLOYMENT — DEC 01 2025
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="ML Support Brain",
    page_icon="rocket",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# MODEL LOADING (NEW — Streamlit Cloud Safe)
# -------------------------------------------------------
@st.cache_resource
def load_models():
    repo_id = "FredaErins/support-triage-models"

    # download from Hugging Face automatically (no timeouts)
    path_type = hf_hub_download(repo_id=repo_id, filename="ticket_type_classifier_PROD_compressed.pkl")
    path_priority = hf_hub_download(repo_id=repo_id, filename="priority_classifier_PROD_compressed.pkl")
    path_queue = hf_hub_download(repo_id=repo_id, filename="queue_routing_classifier_PROD_compressed.pkl")

    return (
        joblib.load(path_type),
        joblib.load(path_priority),
        joblib.load(path_queue)
    )

model_type, model_priority, model_queue = load_models()

# -------------------------------------------------------
# CLEANING FUNCTION
# -------------------------------------------------------
def clean_text(t):
    import re
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop_words = {
        "a","an","the","and","or","is","are","was","were","in","on","at","to",
        "for","with","of","this","that","these","those","i","you","he","she",
        "it","we","they","my","your","his","her","its","our","their","from",
        "as","by","be","been","am","will","can","do","does","did","have",
        "has","had","not","but","if","then","so","no","yes"
    }
    return " ".join(w for w in t.split() if w not in stop_words)

# -------------------------------------------------------
# PREDICTION PIPELINE
# -------------------------------------------------------
def predict_ticket(subject="", body="", queue="", th_priority=0.80, th_queue=0.85):
    text = clean_text(subject + " " + body)

    df = pd.DataFrame([{
        'text': text,
        'queue': str(queue).strip() if queue else "General",
        'priority': "Medium"
    }])

    # Type
    ticket_type = model_type.predict(df[['text', 'queue', 'priority']])[0]
    type_conf = model_type.predict_proba(df[['text', 'queue', 'priority']])[0].max()

    # Priority
    df['ticket_type'] = ticket_type
    priority_input = df[['text', 'queue', 'ticket_type']]
    priority = model_priority.predict(priority_input)[0]
    priority_conf = model_priority.predict_proba(priority_input)[0].max()

    # Queue
    queue_input = pd.DataFrame([{
        'text': text,
        'ticket_type': ticket_type,
        'priority': priority
    }])
    pred_queue = model_queue.predict(queue_input[['text', 'ticket_type', 'priority']])[0]
    queue_conf = model_queue.predict_proba(queue_input)[0].max()

    # Auto logic
    auto_priority = priority if priority_conf >= th_priority else None
    auto_queue = pred_queue if queue_conf >= th_queue else None

    # Final Action
    final_action = (
        "FULLY AUTO-TRIAGED → No human needed" if auto_priority and auto_queue else
        "AUTO-ROUTED → Agent only confirms priority" if auto_queue else
        "AUTO-PRIORITY → Agent confirms queue" if auto_priority else
        "AUTO-TYPE ONLY → Agent decides priority & queue" if type_conf >= 0.90 else
        "HUMAN REVIEW SUGGESTED → Low confidence"
    )

    return {
        "ticket_type": ticket_type,
        "type_confidence": float(type_conf),
        "predicted_priority": priority,
        "priority_confidence": float(priority_conf),
        "auto_set_priority": auto_priority,
        "predicted_queue": pred_queue,
        "queue_confidence": float(queue_conf),
        "auto_route_to": auto_queue,
        "final_action": final_action
    }

# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------
LOG_FILE = "data/prediction_log.csv"
if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp","subject","ticket_type","priority","queue","auto_queue","action"
    ]).to_csv(LOG_FILE, index=False)

def log_prediction(subject, result):
    log_df = pd.read_csv(LOG_FILE)
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject[:100],
        "ticket_type": result["ticket_type"],
        "priority": result["auto_set_priority"] or result["predicted_priority"],
        "queue": result["auto_route_to"] or result["predicted_queue"],
        "auto_queue": bool(result["auto_route_to"]),
        "action": result["final_action"]
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

# -------------------------------------------------------
# UI
# -------------------------------------------------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/skype/289/rocket_1f680.png", width=100)
    st.title("ML Support Brain")
    st.caption("Type → Priority → Queue • 88.8% accuracy")

    log_df = pd.read_csv(LOG_FILE)
    total = len(log_df)
    st.metric("Tickets Processed", total)
    if total > 0:
        st.metric("Auto-Routed Rate", f"{(log_df.auto_queue.sum()/total*100):.1f}%")

    st.divider()
    st.session_state.th_p = st.slider("Auto-Priority Threshold", 0.50, 1.00, 0.80, 0.01)
    st.session_state.th_q = st.slider("Auto-Queue Threshold", 0.50, 1.00, 0.85, 0.01)

st.title("Live Support Ticket Auto-Triage Engine")
st.markdown("**The smartest, safest support AI ever built**")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### **Subject** <span style='color:red'>*</span>", unsafe_allow_html=True)
    subject = st.text_input("", placeholder="e.g. Can't login", key="subject", label_visibility="collapsed")

    st.markdown("### **Body** <span style='color:red'>*</span>", unsafe_allow_html=True)
    body = st.text_area("", height=200, placeholder="Paste full message...", key="body", label_visibility="collapsed")

with col2:
    queue_hint = st.text_input("Current Queue (optional)", "")

if st.button("TRIAGE THIS TICKET", type="primary", use_container_width=True):
    with st.spinner("Thinking..."):
        result = predict_ticket(subject, body, queue_hint, st.session_state.th_p, st.session_state.th_q)
        log_prediction(subject, result)

    st.success("Done!")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Type", result["ticket_type"], f"{result['type_confidence']:.1%}")
    with c2:
        st.metric("Priority", (result["auto_set_priority"] or result["predicted_priority"]).upper(),
                  f"{result['priority_confidence']:.1%}")
    with c3:
        st.metric("Queue", result["auto_route_to"] or result["predicted_queue"],
                  f"{result['queue_confidence']:.1%}")

    st.markdown(f"## {result['final_action']}")
    if result["auto_route_to"]:
        st.balloons()

st.caption("Built solo in 3.5 weeks • Production-ready today")
