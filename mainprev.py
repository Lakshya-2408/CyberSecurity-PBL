# app.py

import streamlit as st
import joblib
import numpy as np
import re
from email import policy
from email.parser import BytesParser

# --- Load Model ---
model = joblib.load("phishing_model.pkl")
tfidf = joblib.load("vectorizer.pkl")

# --- Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_suspicious_url(text):
    trusted_domains = ['forms.gle', 'docs.google.com', 'microsoft.com', 'zoom.us', 'github.com']
    urls = re.findall(r'https?://[^\s]+', text)
    for url in urls:
        if any(trusted in url for trusted in trusted_domains):
            continue  # skip trusted ones
        return 1  # suspicious URL found
    return 0  # all URLs are trusted or no URLs


def parse_eml(file):
    msg = BytesParser(policy=policy.default).parse(file)
    headers = dict(msg.items())
    body = msg.get_body(preferencelist=('plain', 'html'))
    body_text = body.get_content() if body else ""
    return headers, body_text

def extract_header_features(headers):
    from_addr = headers.get("From", "")
    return_path = headers.get("Return-Path", "")

    spf_pass = "pass" in headers.get("Received-SPF", "").lower()
    dkim_pass = "pass" in headers.get("Authentication-Results", "").lower()

    from_dom = from_addr.split("@")[-1].strip().lower() if "@" in from_addr else ""
    return_dom = return_path.split("@")[-1].strip().lower() if "@" in return_path else ""

    trusted_return_domains = [
        "amazonses.com", "sendgrid.net", "mailgun.org", "google.com"
    ]

    domain_mismatch = (
        from_dom != return_dom and return_dom not in trusted_return_domains
    )

    return from_addr, return_path, int(spf_pass), int(dkim_pass), int(domain_mismatch)


def extract_features(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned]).toarray()
    url_feat = has_suspicious_url(text)
    char_count = len(text)
    word_count = len(text.split())
    return np.hstack((vec, [[url_feat, char_count, word_count]]))

# --- Streamlit UI ---
st.set_page_config(page_title="Email Phishing Detector", page_icon="📧")
st.title("📧 Smart Email Phishing Detection")

mode = st.radio("Choose input method:", ["Paste Email Text", "Upload .eml File"])

if mode == "Paste Email Text":
    email_content = st.text_area("📩 Paste Email Body", height=200)
    if st.button("🔍 Detect"):
        if not email_content.strip():
            st.warning("Please paste email content.")
        else:
            features = extract_features(email_content)
            prob = model.predict_proba(features)[0][1]

            st.write(f"**Confidence Score:** `{prob:.2f}`")
            if prob > 0.4:
                st.error("⚠️ Phishing Email Detected!")
            else:
                st.success("✅ Legitimate Email")

elif mode == "Upload .eml File":
    file = st.file_uploader("📁 Upload .eml File", type=["eml"])
    if file and st.button("🔍 Analyze"):
        headers, body = parse_eml(file)
        features = extract_features(body)
        prob = model.predict_proba(features)[0][1]

        st.subheader("📊 Detection Result")
        st.write(f"**Confidence Score:** `{prob:.2f}`")
        if prob > 0.7:
            st.error("⚠️ Phishing Email Detected!")
        else:
            st.success("✅ Legitimate Email")

        from_addr, return_path, spf_pass, dkim_pass, domain_mismatch = extract_header_features(headers)
        st.subheader("🔎 Header Analysis")
        st.write(f"**From:** {from_addr}")
        st.write(f"**Return-Path:** {return_path}")
        st.write(f"**SPF Passed:** {'✅' if spf_pass else '❌'}")
        st.write(f"**DKIM Passed:** {'✅' if dkim_pass else '❌'}")
        st.write(f"**Domain Mismatch:** {'⚠️ Yes' if domain_mismatch else '✅ No'}")
