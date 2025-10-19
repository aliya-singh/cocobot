import streamlit as st
import logging
from config.config import Config
from models.groq_client import GroqClient
from utils.document_handler import DocumentHandler
from utils.search import DocumentSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Companion", page_icon="⚙️", layout="wide")

# Initialize
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    response_mode = st.radio("Mode", ["concise", "detailed"])
    temperature = Config.DEFAULT_TEMPERATURE_CONCISE if response_mode == "concise" else Config.DEFAULT_TEMPERATURE_DETAILED
    temperature = st.slider("Temperature", 0.0, 1.0, temperature)
    
    st.subheader("📚 RAG - Add Knowledge")
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📄 Files", "🌐 Website", "📋 Text"])
    
    with tab1:
        st.write("**Upload Documents**")
        uploaded_files = st.file_uploader(
            "Upload files (TXT, MD, PDF)",
            type=Config.ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files and st.button("📤 Index Files", key="index_files"):
            for file in uploaded_files:
                try:
                    text, success = DocumentHandler.extract_from_file(file, file.type)
                    if success and text:
                        st.session_state.documents[file.name] = text
                        st.success(f"✅ Added: {file.name}")
                    else:
                        st.error(f"No text extracted from {file.name}")
                except Exception as e:
                    st.error(f"Error with {file.name}: {e}")
    
    with tab2:
        st.write("**Add Website Content**")
        website_url = st.text_input(
            "Enter website URL",
            placeholder="https://example.com",
            key="website_url"
        )
        
        if website_url and st.button("🌐 Fetch Website", key="fetch_website"):
            try:
                domain = DocumentHandler.get_domain_name(website_url)
                with st.spinner(f"Fetching {domain}..."):
                    text, success = DocumentHandler.fetch_website_content(website_url)
                    if success and text:
                        st.session_state.documents[domain] = text
                        st.success(f"✅ Added: {domain}")
                    else:
                        st.error("No content extracted")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab3:
        st.write("**Paste Text Content**")
        text_input = st.text_area(
            "Paste content here",
            height=150,
            key="text_input"
        )
        
        text_name = st.text_input(
            "Name this content",
            placeholder="e.g., My Notes",
            key="text_name"
        )
        
        if text_input and text_name and st.button("📝 Add Text", key="add_text"):
            st.session_state.documents[text_name] = text_input[:Config.MAX_DOCUMENT_SIZE]
            st.success(f"✅ Added: {text_name}")
    
    # Show indexed documents
    st.subheader("📚 Indexed Sources")
    if st.session_state.documents:
        if len(st.session_state.documents) > Config.MAX_DOCUMENTS:
            st.warning(f"Max {Config.MAX_DOCUMENTS} documents allowed")
        
        for doc_name in st.session_state.documents.keys():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc_name[:30]}")
            with col2:
                if st.button("❌", key=f"delete_{doc_name}"):
                    del st.session_state.documents[doc_name]
                    st.rerun()
    else:
        st.caption("No sources indexed yet")
    
    st.metric("Total Sources", len(st.session_state.documents))
    st.metric("Queries", st.session_state.query_count)

# Main
st.title("🔧 AI Knowledge Companion")
st.write("Ask questions about your documents, websites, or anything technical.")

if st.session_state.documents:
    with st.expander(f"📚 {len(st.session_state.documents)} Sources Active"):
        for doc_name in st.session_state.documents.keys():
            st.caption(f"✅ {doc_name}")

# Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize Groq client
                client = GroqClient()
                
                # Build context from documents
                context = DocumentSearch.build_rag_context(user_input, st.session_state.documents)
                
                # Build system prompt
                system = "Be concise and practical." if response_mode == "concise" else "Be detailed, thorough, and provide examples."
                
                # Build messages
                messages = [
                    {"role": "system", "content": f"{system}\n\n{context}"},
                    *st.session_state.chat_history[-5:]
                ]
                
                # Get response
                answer = client.chat(messages, temperature=temperature)
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.query_count += 1
                
                # Show sources used
                if st.session_state.documents:
                    with st.expander("📚 Sources"):
                        for doc_name in st.session_state.documents.keys():
                            st.caption(f"✅ {doc_name}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💬 Queries", st.session_state.query_count)
with col2:
    st.metric("📚 Sources", len(st.session_state.documents))
with col3:
    st.metric("📝 History", len(st.session_state.chat_history))