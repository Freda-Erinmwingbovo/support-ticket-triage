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
        p_type = hf_hub_download(
            repo_id=repo,
            filename="ticket_type_classifier_PROD_compressed.pkl",
            repo_type="dataset"
        )
        p_priority = hf_hub_download(
            repo_id=repo,
            filename="priority_classifier_PROD_compressed.pkl",
            repo_type="dataset"
        )
        p_queue = hf_hub_download(
            repo_id=repo,
            filename="queue_routing_PROD_compressed.pkl",
            repo_type="dataset"
        )

    except Exception as e:
        st.error(f"❌ Failed to download models: {e}")
        st.stop()

    return (
        joblib.load(p_type),
        joblib.load(p_priority),
        joblib.load(p_queue)
    )



model_type, model_priority, model_queue = load_models()


# ========================= TEXT CLEANING & PREDICTION =========================
def clean_text(t):
    import re
    if pd.isna(t): return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop_words = {"a","an","the","and","or","is","are","was","were","in","on","at","to","for","with",
                  "of","this","that","these","those","i","you","he","she","it","we","they","my","your",
                  "his","her","its","our","their","from","as","by","be","been","am","will","can","do",
                  "does","did","have","has","had","not","but","if","then","so","no","yes"}
    return " ".join(w for w in t.split() if w not in stop_words)

def predict_ticket(subject="", body="", queue="", th_priority=0.80, th_queue=0.85):
    text = clean_text(subject + " " + body)

    df = pd.DataFrame([{
        'text': text,
        'queue': str(queue).strip() if queue else "General",
        'priority': "Medium"
    }])

    # Model 1
    ticket_type = model_type.predict(df[['text', 'queue', 'priority']])[0]
    type_conf = model_type.predict_proba(df[['text', 'queue', 'priority']])[0].max()
    df['ticket_type'] = ticket_type

    # Model 2
    priority_input = df[['text', 'queue', 'ticket_type']]
    priority = model_priority.predict(priority_input)[0]
    priority_conf = model_priority.predict_proba(priority_input)[0].max()

    # Model 3
    queue_input = pd.DataFrame([{'text': text, 'ticket_type': ticket_type, 'priority': priority}])
    pred_queue = model_queue.predict(queue_input[['text', 'ticket_type', 'priority']])[0]
    queue_conf = model_queue.predict_proba(queue_input)[0].max()

    auto_priority = priority if priority_conf >= th_priority else None
    auto_queue = pred_queue if queue_conf >= th_queue else None

    if auto_priority and auto_queue:
        final_action = "FULLY AUTO-TRIAGED → No human needed"
    elif auto_queue:
        final_action = "AUTO-ROUTED → Agent only confirms priority"
    elif auto_priority:
        final_action = "AUTO-PRIORITY → Agent confirms queue"
    elif type_conf >= 0.90:
        final_action = "AUTO-TYPE ONLY → Agent decides priority & queue"
    else:
        final_action = "HUMAN REVIEW SUGGESTED → Low overall confidence"

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

# ========================= LOGGING =========================
LOG_FILE = "data/prediction_log.csv"
if not os.path.exists("data"): os.makedirs("data")
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp","subject","ticket_type","priority","queue","auto_queue","action"]).to_csv(LOG_FILE, index=False)

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

# ========================= SIDEBAR =========================
with st.sidebar:
    st.image("https://em-content.zobj.net/source/skype/289/rocket_1f680.png", width=100)
    st.title("ML Support Brain")
    st.caption("Type → Priority → Queue • 88.8% accuracy")

    log_df = pd.read_csv(LOG_FILE)
    total = len(log_df)
    st.metric("Tickets Processed", total)
    if total > 0:
        auto_rate = (log_df["auto_queue"].sum() / total) * 100
        st.metric("Auto-Routed Rate", f"{auto_rate:.1f}%")

    st.divider()
    st.info("*Global Auto-Action Thresholds*\n• Only act when confidence ≥ these values\n• Lower = more automation\n• Current = ultra-safe (~30%)")
    st.session_state.th_p = st.slider("Auto-Priority Threshold", 0.50, 1.00, 0.80, 0.01)
    st.session_state.th_q = st.slider("Auto-Queue Threshold",    0.50, 1.00, 0.85, 0.01)

# ========================= MAIN UI =========================
st.title("Live Support Ticket Auto-Triage Engine")
st.markdown("*The smartest, safest support AI ever built*")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### *Subject* <span style='color:red'>*</span>", unsafe_allow_html=True)
    subject = st.text_input("", placeholder="e.g. Can't login after update", key="subject", label_visibility="collapsed")
    if not subject: st.error("Subject is required")
    st.markdown("### *Body* <span style='color:red'>*</span>", unsafe_allow_html=True)
    body = st.text_area("", height=200, placeholder="Paste the full customer message here...", key="body", label_visibility="collapsed")
    if not body: st.error("Body is required")

with col2:
    queue_hint = st.text_input("Current Queue (optional)", placeholder="e.g. billing, technical")

# ========================= TRIAGE =========================
if st.button("TRIAGE THIS TICKET", type="primary", use_container_width=True, disabled=not (subject and body)):
    with st.spinner("Analyzing ticket..."):
        result = predict_ticket(subject, body, queue_hint,
                                st.session_state.th_p, st.session_state.th_q)
        log_prediction(subject, result)

    st.success("Triage Complete!")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ticket Type", result["ticket_type"], f"{result['type_confidence']:.1%}")
    with c2:
        pri = (result["auto_set_priority"] or result["predicted_priority"] or "Unknown").upper()
        st.metric("Priority", pri, f"{result['priority_confidence']:.1%}")
    with c3:
        q = result["auto_route_to"] or result["predicted_queue"]
        st.metric("Queue", q, f"{result['queue_confidence']:.1%}")

    st.markdown(f"## {result['final_action']}")
    if result["auto_route_to"]:
        st.balloons()
        st.success(f"AUTO-ROUTED TO → *{result['auto_route_to']}*")
    else:
        st.warning("No auto-routing — model is not confident enough")

st.caption("Built solo in 3.5 weeks • Safer & smarter than Zendesk AI • Production-ready today")













