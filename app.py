import hashlib
import streamlit as st

from pdf_utils import read_pdf, chunk_text
from rag import embed_chunks, ask_pdf
from config import GEMINI_API_KEY

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PDF Intelligence | RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Metric Cards in Sidebar */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(229, 231, 235, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4F46E5;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Source Pill Badge */
    .source-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
        border: 1px solid #C7D2FE;
    }
    
    /* Clean Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "embedded_docs" not in st.session_state:
    st.session_state.embedded_docs = None

if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = None


# Helper to compute file hash
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()


# ---------------------------------------------------------
# Sidebar & Document Upload
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Document Center")
    
    # API Key check status
    if GEMINI_API_KEY:
        st.success("⚡ Gemini API Key Active", icon="✅")
    else:
        st.error("⚠️ GEMINI_API_KEY missing (add in Secrets or .env)", icon="🚨")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload any text-based PDF to analyze and ask questions."
    )
    
    st.divider()

    # Model / RAG Parameters
    st.markdown("### ⚙️ Search Settings")
    top_k = st.slider(
        "Context Excerpts (Top-K)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of relevant PDF chunks passed to Gemini for generating the answer."
    )

    # Document Metrics (Display if file is indexed)
    if st.session_state.pdf_metadata:
        st.divider()
        st.markdown("### 📊 Document Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Pages", st.session_state.pdf_metadata.get("num_pages", 0))
            st.metric("Total Chunks", st.session_state.pdf_metadata.get("num_chunks", 0))
        with col2:
            st.metric("Total Words", f"{st.session_state.pdf_metadata.get('word_count', 0):,}")
            
    st.divider()
    
    # Action buttons
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
# Hero Header Banner
st.markdown("""
<div class="hero-header">
    <div class="hero-title">📄 PDF Intelligence Chatbot</div>
    <div class="hero-subtitle">Ask questions, extract insights, and analyze documents in seconds using RAG & Gemini AI</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Document Processing Logic (Cached Embeddings)
# ---------------------------------------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = get_file_hash(file_bytes)

    # If new file is uploaded, process and embed chunks
    if st.session_state.file_hash != current_hash:
        with st.status("🚀 Processing & Indexing PDF...", expanded=True) as status:
            st.write("📖 Reading PDF content...")
            text, meta = read_pdf(uploaded_file)
            
            if not text.strip():
                status.update(label="❌ PDF is empty or contains no extractable text.", state="error")
                st.error("Could not extract any text from this PDF. It might be scanned or image-only.")
                st.stop()
                
            st.write("✂️ Splitting document into search chunks...")
            chunks = chunk_text(text)
            
            st.write(f"🧠 Generating Vector Embeddings for {len(chunks)} chunks...")
            
            progress_bar = st.progress(0)
            
            def update_progress(current, total):
                progress_bar.progress(current / total)

            try:
                embedded_docs = embed_chunks(chunks, progress_callback=update_progress)
                
                meta["num_chunks"] = len(chunks)
                st.session_state.file_hash = current_hash
                st.session_state.embedded_docs = embedded_docs
                st.session_state.pdf_metadata = meta
                st.session_state.messages = []  # Reset chat history for new doc
                
                status.update(label="✅ Document successfully indexed!", state="complete")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error generating embeddings", state="error")
                st.error(f"Error: {e}")
                st.stop()

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------
if not st.session_state.embedded_docs:
    st.info("👈 Please upload a PDF document from the sidebar to start asking questions.")
else:
    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source excerpts if available for assistant responses
            if message.get("sources"):
                with st.expander("📚 View Relevant PDF Excerpts & Relevance Scores"):
                    for idx, source in enumerate(message["sources"], start=1):
                        score_pct = source["score"] * 100
                        st.markdown(f"""
                        <span class="source-badge">Excerpt #{idx} | Similarity Match: {score_pct:.1f}%</span>
                        """, unsafe_allow_html=True)
                        st.caption(source["text"])
                        if idx < len(message["sources"]):
                            st.divider()

    # Chat Input for User
    if user_query := st.chat_input("Ask a question about your PDF..."):
        if not user_query.strip():
            st.warning("Please enter a valid question.")
        else:
            # Render user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching PDF and generating answer..."):
                    try:
                        answer, sources = ask_pdf(
                            user_query,
                            st.session_state.embedded_docs,
                            top_k=top_k
                        )
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 View Relevant PDF Excerpts & Relevance Scores"):
                                for idx, source in enumerate(sources, start=1):
                                    score_pct = source["score"] * 100
                                    st.markdown(f"""
                                    <span class="source-badge">Excerpt #{idx} | Similarity Match: {score_pct:.1f}%</span>
                                    """, unsafe_allow_html=True)
                                    st.caption(source["text"])
                                    if idx < len(sources):
                                        st.divider()
                                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        error_msg = f"❌ An error occurred while generating the answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })