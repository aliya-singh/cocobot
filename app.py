import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse

st.set_page_config(page_title="AI Companion", page_icon="⚙️", layout="wide")

# Initialize
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

def extract_pdf_text(file):
    """Extract text from PDF"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text[:3000]
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

def fetch_website_content(url):
    """Fetch and extract text from website"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:3000]
    except Exception as e:
        st.error(f"Website Error: {e}")
        return ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    response_mode = st.radio("Mode", ["concise", "detailed"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3 if response_mode == "concise" else 0.7)
    
    st.subheader("📚 RAG - Add Knowledge")
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📄 Files", "🌐 Website", "📋 Text"])
    
    with tab1:
        st.write("**Upload Documents**")
        uploaded_files = st.file_uploader(
            "Upload files (TXT, MD, PDF)",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files and st.button("📤 Index Files", key="index_files"):
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        text = extract_pdf_text(file)
                    else:
                        text = file.getvalue().decode('utf-8')
                    
                    if text:
                        st.session_state.documents[file.name] = text
                        st.success(f"✅ Added: {file.name}")
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
            if not website_url.startswith(('http://', 'https://')):
                website_url = 'https://' + website_url
            
            try:
                domain = urlparse(website_url).netloc
                with st.spinner(f"Fetching {domain}..."):
                    text = fetch_website_content(website_url)
                    if text:
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
            st.session_state.documents[text_name] = text_input[:3000]
            st.success(f"✅ Added: {text_name}")
    
    # Show indexed documents
    st.subheader("📚 Indexed Sources")
    if st.session_state.documents:
        for doc_name in st.session_state.documents.keys():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc_name}")
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
                api_key = st.secrets.get("GROQ_API_KEY")
                if not api_key:
                    st.error("❌ No Groq API key in secrets")
                    st.stop()
                
                # Build context from all sources
                context = ""
                if st.session_state.documents:
                    context = "KNOWLEDGE BASE:\n"
                    for doc_name, doc_text in st.session_state.documents.items():
                        # Search for relevant content
                        query_words = user_input.lower().split()
                        if any(word in doc_text.lower() for word in query_words):
                            context += f"\n[{doc_name}]\n{doc_text}\n"
                    
                    if context == "KNOWLEDGE BASE:\n":
                        context = "No relevant documents found in knowledge base.\n"
                
                system = "Be concise and practical." if response_mode == "concise" else "Be detailed, thorough, and provide examples."
                
                # Direct Groq API REST call
                url = "https://api.groq.com/openai/v1/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                messages = [
                    {"role": "system", "content": f"{system}\n\n{context}"},
                    *st.session_state.chat_history[-5:]
                ]
                
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
                
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['choices'][0]['message']['content']
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.query_count += 1
                    
                    # Show sources used
                    with st.expander("📚 Sources Used"):
                        if st.session_state.documents:
                            for doc_name in st.session_state.documents.keys():
                                st.caption(f"✅ {doc_name}")
                        else:
                            st.caption("No sources")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💬 Queries", st.session_state.query_count)
with col2:
    st.metric("📚 Sources", len(st.session_state.documents))
with col3:
    st.metric("📝 Chat History", len(st.session_state.chat_history))