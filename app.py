import streamlit as st
import pickle
from groq import Groq

st.set_page_config(page_title="Review Analyzer", page_icon="💬")
st.title("💬 Customer Review Analyzer")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.header("🤖 Select Model")
    selected_model = st.selectbox("Groq model:", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

# ── Load trained model & vectorizer ──────────
@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    vectorizer, model = load_model()
    st.sidebar.success("✅ ML Model loaded!")
except FileNotFoundError:
    st.sidebar.error("❌ model.pkl or vectorizer.pkl not found.")
    st.stop()

# ── Issue classifier ──────────────────────────
def get_issue(text):
    t = text.lower()
    if any(w in t for w in ["refund", "money back", "charge"]):
        return "Refund"
    if any(w in t for w in ["delivery", "shipping", "arrived", "late", "delay"]):
        return "Delivery"
    if any(w in t for w in ["broken", "damaged", "defective", "not working", "faulty"]):
        return "Product Issue"
    if any(w in t for w in ["price", "expensive", "overpriced"]):
        return "Pricing"
    if any(w in t for w in ["service", "support", "staff", "rude"]):
        return "Customer Service"
    return "General"

# ── Priority ──────────────────────────────────
def get_priority(sentiment, text):
    t = text.lower()
    if sentiment == "Negative" and any(w in t for w in ["refund", "fraud", "legal", "lawsuit"]):
        return "High"
    if sentiment == "Negative":
        return "Medium"
    return "Low"

# ── Groq reply generator ──────────────────────
def refine_reply(review, analysis, api_key, model_name):
    client = Groq(api_key=api_key)
    prompt = f"""
You are a customer support agent.

Customer review:
"{review}"

Analysis:
- Sentiment: {analysis['sentiment']}
- Issue: {analysis['issue']}
- Priority: {analysis['priority']}

Write a polite, human-like customer support reply (2-3 lines).
Rules:
- Start with: Dear Customer,
- Use "we" throughout
- End with: Warm regards, Support Team
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ── Full pipeline ─────────────────────────────
def process_review(text, api_key, model_name):
    vec       = vectorizer.transform([text])
    sentiment = model.predict(vec)[0]
    issue     = get_issue(text)
    priority  = get_priority(sentiment, text)

    analysis = {
        "sentiment": sentiment,
        "issue":     issue,
        "priority":  priority
    }

    reply = refine_reply(text, analysis, api_key, model_name)

    return {
        "review":    text,
        "sentiment": sentiment,
        "issue":     issue,
        "priority":  priority,
        "reply":     reply
    }

# ── Main ──────────────────────────────────────
st.subheader("Enter a review")

review_text = st.text_area("", height=120, placeholder="e.g. Worst purchase of my life, broken on arrival...")

if st.button("Analyze", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("Please enter a review.")
    elif not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = process_review(review_text, groq_api_key, selected_model)

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", result["sentiment"])
                col2.metric("Issue",     result["issue"])
                col3.metric("Priority",  result["priority"])

                st.markdown("---")
                st.subheader("Generated Reply")
                st.info(result["reply"])

                st.download_button(
                    "⬇️ Download reply as .txt",
                    data=result["reply"],
                    file_name="reply.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {e}")