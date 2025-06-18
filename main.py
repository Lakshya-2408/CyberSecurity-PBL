import streamlit as st
import joblib
import numpy as np
import re
from email import policy
from email.parser import BytesParser
from textblob import TextBlob
st.set_page_config(page_title="Phishing Detector", layout="wide")
# Load model and vectorizer
model = joblib.load("phishing_model.pkl")
tfidf = joblib.load("vectorizer.pkl")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    /* Set global font and layout */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f2f4f8;
        color: #333;
    }

    /* Wide layout */
    .main {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e3eaf2;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        color: #333;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d0dbe8;
        color: black;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #145a8d;
    }

    /* Text area */
    textarea {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 10px;
        font-size: 16px;
        color: #222;
    }

    /* Subheaders */
    h3 {
        margin-top: 30px;
        color: #1f77b4;
    }

    /* Alerts */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

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
            continue
        return 1
    return 0

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
    domain_mismatch = (from_dom != return_dom and return_dom not in trusted_return_domains)
    return from_addr, return_path, int(spf_pass), int(dkim_pass), int(domain_mismatch)

def extract_features(text, spf_pass=1, dkim_pass=1, domain_mismatch=0):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned]).toarray()
    url_feat = has_suspicious_url(text)
    char_count = len(text)
    word_count = len(text.split())
    spam_keywords = ['login', 'verify', 'password', 'bank', 'urgent', 'account', 'update', 'click', 'security']
    spam_kw_count = sum(1 for kw in spam_keywords if kw in text.lower())
    sentiment = TextBlob(text).sentiment.polarity

    return np.hstack((vec, [[
        url_feat, char_count, word_count, spam_kw_count,
        sentiment, spf_pass, dkim_pass, domain_mismatch
    ]]))

# --- App Header ---

st.title("📧 Smart Email Phishing Detection System")

# --- Tabs ---
tabs = st.tabs(["📨 Detector", "ℹ️ About", "📥 .eml Instructions"])

# --- Tab 1: Detection ---
with tabs[0]:
    st.subheader("Choose Email Input Mode")
    mode = st.radio("", ["Paste Email Text", "Upload .eml File"], horizontal=True)

    if mode == "Paste Email Text":
        email_content = st.text_area("📩 Paste the email content here:", height=250)
        if st.button("🔍 Detect"):
            if not email_content.strip():
                st.warning("⚠️ Please paste email content.")
            else:
                features = extract_features(email_content)
                prob = model.predict_proba(features)[0][1]

                st.write(f"**Confidence Score:** `{prob:.2f}`")
                if prob > 0.4:
                    st.error("⚠️ Phishing Email Detected!")
                else:
                    st.success("✅ Legitimate Email")

    elif mode == "Upload .eml File":
        file = st.file_uploader("📁 Upload your `.eml` file", type=["eml"])
        if file and st.button("🔍 Analyze"):
            headers, body = parse_eml(file)
            from_addr, return_path, spf_pass, dkim_pass, domain_mismatch = extract_header_features(headers)
            features = extract_features(body, spf_pass, dkim_pass, domain_mismatch)
            prob = model.predict_proba(features)[0][1]

            st.subheader("📊 Detection Result")
            st.write(f"**Confidence Score:** `{prob:.2f}`")
            if prob > 0.7:
                st.error("⚠️ Phishing Email Detected!")
            else:
                st.success("✅ Legitimate Email")

            st.subheader("🔎 Header Analysis")
            st.write(f"**From:** {from_addr}")
            st.write(f"**Return-Path:** {return_path}")
            st.write(f"**SPF Passed:** {'✅' if spf_pass else '❌'}")
            st.write(f"**DKIM Passed:** {'✅' if dkim_pass else '❌'}")
            st.write(f"**Domain Mismatch:** {'⚠️ Yes' if domain_mismatch else '✅ No'}")

# --- Tab 2: About ---
with tabs[1]:
    st.subheader("👨‍💻 About This Project")
    st.markdown("""
    ### 📌 Project Overview
    The **Smart Email Phishing Detection System** is an academic project developed as part of the *Cyber Security* curriculum under **Project-Based Learning (PBL)**. It aims to combat the increasing threat of email phishing attacks through a combination of Machine Learning (ML), Natural Language Processing (NLP), and email forensics.

    Phishing remains one of the most dangerous cyber-attack vectors, tricking users into revealing sensitive data like passwords, credit card information, or login credentials. This project is an effort to automate the identification of such malicious emails and provide an easy-to-use interface for users.

    ---

    ### 🧠 Technologies Used

    - **Natural Language Processing (NLP):**
      - Analyzes the email text to understand its structure and intent.
      - Extracts important features like spam keywords, tone, and sentiment polarity using libraries like **TextBlob**.

    - **Email Header Forensics:**
      - Parses `.eml` files to extract hidden metadata.
      - Checks for SPF and DKIM validation, which are essential indicators of sender legitimacy.
      - Identifies mismatches in the 'From' and 'Return-Path' domains (often used in spoofing).

    - **Machine Learning (Naive Bayes Classifier):**
      - A lightweight, fast, and interpretable model trained on thousands of real and phishing emails.
      - Takes both content-based and technical header features as input to classify the email.

    - **TF-IDF Vectorization:**
      - Transforms raw text into numerical features for the ML model.
      - Captures keyword frequency while minimizing the weight of common but less-informative words.

    - **Streamlit:**
      - Provides a clean and interactive web interface.
      - Allows users to either paste an email manually or upload `.eml` files for deeper inspection.

    ---

    ### 🧪 Key Functionalities

    ✅ Supports **two types of inputs**:
    - Pasted email content (for basic users).
    - Uploaded `.eml` files (for forensic-level inspection).

    ✅ Performs **header-level analysis**:
    - SPF (Sender Policy Framework)
    - DKIM (DomainKeys Identified Mail)
    - Domain Mismatch between `From:` and `Return-Path`

    ✅ Computes a **confidence score** (0 to 1) for prediction:
    - Shows how likely an email is to be phishing based on content and headers.

    ✅ Provides **real-time feedback** with color-coded results and alerts:
    - Green for legitimate emails
    - Red for phishing attempts
    - Yellow warnings for partial header mismatches

    ✅ Offers **complete transparency**:
    - Displays extracted metadata to educate users about email security markers.

    ---

    ### 🎯 Objective
    The goal of this project is not only to create a functional phishing detection system, but also to:
    - Raise awareness about how phishing works.
    - Encourage users to look beyond just the visible content of emails.
    - Apply real-world cybersecurity knowledge in a meaningful project.

    This tool can be used as a **learning platform**, a **basic enterprise prototype**, or a **browser-integrated phishing alert system** in future extensions.

    ---

    ### 👥 Project Team

    - **🔹 Team Leader**:  
      *Anuj Kumar Saroj*

    - **🔹 Team Members**:
        - Vedant Devrani  
        - Priyanshu Gupta  
        - Lakshya Dhiman

    ---

    ### 📚 Academic Context

    - **Course**: Cyber Security  
    - **Project Type**: PBL (Project-Based Learning)  
    - **University**: Graphic Era Hill University Dehradun Campus 
    - **Semester**: 4th Semester (B.Tech, 2nd Year)

    This project reflects both theoretical knowledge and practical implementation in cybersecurity, natural language processing, and software engineering. 

    ---

    ### 🔮 Future Scope

    - Integration with real-time email clients like Gmail/Outlook using APIs.
    - Building a browser plugin to flag phishing emails directly in the inbox.
    - Enhancing the ML model with deep learning and continuous dataset updates.
    - Logging and alert system for organizational IT teams.
    - Threat intelligence integration with public phishing databases (like PhishTank).

    ---
    """, unsafe_allow_html=True)


# --- Tab 3: Instructions ---
with tabs[2]:
    st.subheader("📥 How to Download .eml Files")

    st.markdown("""
    To upload an email for phishing detection, you need to save the email as a `.eml` file format.
    Below are step-by-step instructions for different platforms:

    ---

    ### 📧 **1. Gmail (Web Browser)**
    1. Open the email you want to analyze.
    2. Click on the three vertical dots **(⋮)** at the top-right of the email pane.
    3. Select **"Download message"**.
    4. The email will be downloaded as a `.eml` file to your computer.
    
    💡 *Tip: If it opens in a new tab, right-click and choose “Save As”.*

    """)


    st.markdown("""
    ---

    ### 🖥️ **2. Microsoft Outlook (Desktop App)**

    1. Open Outlook and go to your inbox.
    2. Double-click to open the email you want to save.
    3. Click **File → Save As**.
    4. In the "Save as type" dropdown, select **Outlook Message Format – .eml**.
    5. Save it to your desired folder.

    💡 *Make sure you choose the correct format, not .msg or .txt.*

    """)

    

    st.markdown("""
    ---

    ### 📱 **3. Yahoo Mail & Others**
    - Most providers do **not** offer direct `.eml` downloads.
    - You may need to:
      - Open the email in a desktop client (like Thunderbird or Outlook).
      - Or use “Print” → “Save as PDF” as a fallback, though **this won’t work** for detection.

    ✅ For phishing analysis, only `.eml` format is supported.

    ---

    ### ⚠️ Precautions

    - Only upload **non-sensitive** emails.
    - Do **not** upload personal emails containing:
        - Passwords  
        - Financial details  
        - Legal documents  

    📎 This tool is for **educational and research purposes only**.

    ---
    """)

