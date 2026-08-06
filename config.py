import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = None

# 1. Try reading from Streamlit Secrets (for Cloud Deployment)
try:
    if hasattr(st, "secrets") and st.secrets is not None:
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
except Exception:
    pass

# 2. Fallback to environment variables / .env (for Local Run)
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure value is clean string
if GEMINI_API_KEY:
    GEMINI_API_KEY = str(GEMINI_API_KEY).strip()