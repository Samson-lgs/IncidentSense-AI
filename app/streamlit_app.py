from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="IncidentSense AI", page_icon="🧠", layout="wide")
st.title("🧠 IncidentSense AI")
st.caption("ML-powered incident triage: root cause prediction, anomaly detection, similarity search and drift monitoring.")

with st.sidebar:
    st.subheader("Incident")
    service = st.selectbox("Service", ["checkout", "payments", "identity", "orders", "search", "gateway"])
    environment = st.selectbox("Environment", ["production", "staging", "qa"])
    severity = st.selectbox("Severity", ["INFO", "WARN", "ERROR", "CRITICAL"], index=2)
    error_code = st.text_input("Error code", "DB_TIMEOUT")
    http_status = st.number_input("HTTP status", min_value=100, max_value=599, value=500)
    latency_ms = st.number_input("Latency (ms)", min_value=0.0, value=1500.0)
    log_message = st.text_area("Log message", "database timeout after 30 seconds on orders query")
    analyze = st.button("Analyze incident", type="primary", use_container_width=True)

if analyze:
    payload = {
        "service": service,
        "environment": environment,
        "severity": severity,
        "error_code": error_code,
        "http_status": http_status,
        "latency_ms": latency_ms,
        "log_message": log_message,
    }
    try:
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        pred = result["prediction"]
        anomaly = result["anomaly"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted root cause", pred["root_cause"].title())
        c2.metric("Confidence", f"{pred['confidence']:.1%}")
        c3.metric("Anomaly", "Detected" if anomaly["is_anomaly"] else "Normal")

        left, right = st.columns(2)
        with left:
            st.subheader("Model explanation")
            st.dataframe(pred["top_probabilities"], use_container_width=True, hide_index=True)
        with right:
            st.subheader("Similar incidents")
            st.dataframe(result["similar_incidents"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"API request failed: {exc}")

st.divider()
st.subheader("Model quality")
try:
    metrics = requests.get(f"{API_URL}/metrics", timeout=5).json()
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
    c.metric("Weighted F1", f"{metrics['weighted_f1']:.1%}")
except Exception:
    st.info("Training metrics will appear here after the model is trained.")
