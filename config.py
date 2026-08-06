import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Try reading from Streamlit Secrets first (for Cloud Deployment), fallback to .env (for Local)
GEMINI_API_KEY = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")