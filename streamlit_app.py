import streamlit as st
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG pipelines and LLM engine
from rag_pipeline import VehicleSpecRAG as LocalRAG
from rag_pipeline_API import VehicleSpecRAG as ApiRAG
import llm_engine

# Configuration and Styling
st.set_page_config(
    page_title="Spercer | Advanced Specs RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamic CSS injection
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Inter:wght@300;400;600&family=Rajdhani:wght@500;700&display=swap');

    /* ANIMATED BACKGROUND */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #050505, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }

    /* HEADERS & TITLES */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #fff !important;
        text-shadow: 0 0 10px rgba(0, 242, 96, 0.8), 0 0 20px rgba(0, 242, 96, 0.4);
    }
    
    h1 {
        background: linear-gradient(90deg, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
    }

    /* GLASSMORPHISM SIDEBAR - INCREASED CONTRAST */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.85); /* Darker background for better contrast */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 242, 96, 0.2);
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    /* FORCE SIDEBAR TEXT COLOR */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.8);
    }
    
    /* RADIO BUTTONS */
    .stRadio > label {
        color: #00f260 !important;
        font-weight: bold;
        font-size: 1.1em;
    }

    /* CUSTOM INPUT FIELDS */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(0, 242, 96, 0.3) !important;
        color: #00f260 !important;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Inter', sans-serif;
        font-size: 1.1em;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00f260 !important;
        box-shadow: 0 0 20px rgba(0, 242, 96, 0.3) !important;
        background: rgba(0, 0, 0, 0.6) !important;
    }

    /* CHAT MESSAGES - NEON STYLE */
    .chat-message-user {
        background: linear-gradient(135deg, rgba(5, 117, 230, 0.1), rgba(0, 0, 0, 0.4));
        border: 1px solid rgba(5, 117, 230, 0.5);
        border-right: 4px solid #0575E6;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        text-align: right;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .chat-message-bot {
        background: linear-gradient(135deg, rgba(0, 242, 96, 0.05), rgba(0, 0, 0, 0.4));
        border: 1px solid rgba(0, 242, 96, 0.4);
        border-left: 4px solid #00f260;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        position: relative;
    }

    /* INFO/WARNING BOXES */
    .stAlert {
        background: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #ffcc00 !important;
        color: #fff !important;
    }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(90deg, #0575E6, #00f260) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(5, 117, 230, 0.4) !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(0, 242, 96, 0.6) !important;
        color: #fff !important;
    }
    
    /* JSON OUTPUT */
    div[data-testid="stJson"] {
        background: rgba(0, 20, 0, 0.4) !important;
        border: 1px solid rgba(0, 242, 96, 0.2);
        border-radius: 10px;
        font-family: 'Courier New', monospace;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #0575E6, #00f260); 
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "API"

# Helper Functions
def initialize_rag(mode: str):
    """Initializes the RAG system based on the selected mode."""
    pdf_path = os.path.join("PDFs", "Assignment-specs-extraction 1.pdf")
    hf_api_key = os.getenv("HF_API_KEY")

    if not hf_api_key:
        st.error("❌ `HF_API_KEY` not found in environment variables.")
        st.stop()

    with st.spinner(f"⚡ Powering up {mode} Neural Core..."):
        if mode == "API":
            rag = ApiRAG(pdf_path, hf_api_key)
        else:
            rag = LocalRAG(pdf_path, hf_api_key)

        # Check/Build Index
        if rag.check_index_exists():
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            embeddings = HuggingFaceEmbeddings(model_name=rag.embedding_model_name)
            rag.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=rag.persist_directory,
            )
            rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 15})
        else:
            # Rebuild index
            st.info("⚠️ Index missing. Constructing Vector Matrix...")
            rag.convert_pdf_to_markdown()
            rag.create_chunks()
            rag.initialize_retriever(k=15)
            
        st.session_state.rag_system = rag
        st.session_state.current_mode = mode
        st.toast(f"✅ SPIRIT Online: {mode} Mode Active", icon="⚡")
        time.sleep(1)
        st.rerun()

# UI Layout

# Sidebar
with st.sidebar:
    st.title("⚡ SPERCER")
    st.caption("Advanced Spec Extraction System")
    st.markdown("---")
    
    st.markdown("### 🎛️ COMPUTE CORE")
    
    # Mode Selection
    selected_mode = st.radio(
        "Compute Node:",
        ["API", "Local"],
        captions=["API Based", "Local GPU based"],
        index=0 if st.session_state.current_mode == "API" else 1,
        label_visibility="collapsed"
    )

    # WARNINGS LOGIC
    if selected_mode == "API":
        st.warning("""
        **⚠️ QUANTIZED MODEL DETECTED**
        
        Using free-tier hosting API. Model is heavily quantized. 
        **Precision**: Low (Int8/NF4 typically).
        **Risk**: Potential loss of nuance in complex extractions.
        """)
    else:
        st.error("""
        **⚠️ HIGH PERFORMANCE COMPUTE REQ**
        
        **RUNNING FLOAT16 NATIVE PRECISION.**
        Ensure you have sufficient VRAM (16GB+ recommended). 
        Accuracy is maximized but requires powerful local hardware.
        """)
    
    # Re-initialize if mode changes
    if selected_mode != st.session_state.current_mode:
        if st.button("🔄 Initiate Switch Sequence"):
            st.session_state.rag_system = None
            initialize_rag(selected_mode)

    st.markdown("---")
    st.markdown("### 📊 Telemetry")
    status_text = "🟢 ONLINE" if st.session_state.rag_system else "🔴 OFFLINE"
    st.markdown(f"Status: **{status_text}**")
    st.markdown(f"Mode: **{st.session_state.current_mode.upper()}**")
    
    if st.button("🧹 Purge Memory"):
        st.session_state.messages = []
        st.rerun()

# Main Content
st.title("SPERCER")
st.markdown("#### *Automotive Intelligence & Specification Retrieval*")
st.divider()

# Initial Load check
if st.session_state.rag_system is None:
    initialize_rag(st.session_state.current_mode)

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class='chat-message-user'>
            <div style='font-size: 0.8em; opacity: 0.8; margin-bottom: 5px;'>COMMANDER</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-message-bot'>
            <div style='font-size: 0.8em; opacity: 0.8; margin-bottom: 5px; color: #00f260;'>SPERCER AI</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
        if "data" in message and message["data"]:
            st.json(message["data"])

# Chat Input
query = st.chat_input("Enter specification query parameters...")

if query:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # 2. Process
    with st.spinner("⚡ SPERCER IS THINKING..."):
        try:
            rag = st.session_state.rag_system
            
            # Retrieve
            docs = rag.retrieve_with_references(query)
            
            if not docs:
                response_text = "❌ ANALYSIS FAILED. No relevant documents located in vector matrix."
                st.session_state.messages.append({"role": "assistant", "content": response_text, "data": []})
            else:
                top_docs = docs[:2]
                context = "\n---\n".join([d.page_content for d in top_docs])
                
                # Extract
                if st.session_state.current_mode == "API":
                    data = llm_engine.extract_with_llama_api(context, query)
                else:
                    data = llm_engine.extract_with_llama_local(context, query)
                
                if data:
                    pages = sorted({d.metadata.get("page", 0) for d in top_docs})
                    response_text = f"✅ DATA EXTRACTION COMPLETE. Located {len(data)} spec points (Pages: {pages})."
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "data": data})
                else:
                    response_text = "⚠️ CONTEXT LOCATED BUT EXTRACTION INCONCLUSIVE. No structured data found."
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "data": []})
        
        except Exception as e:
            st.error(f"SYSTEM FAILURE: {str(e)}")
    
    st.rerun()

