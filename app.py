import streamlit as st
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Knowledge Companion", page_icon="⚙️", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

def extract_text_from_file(file):
    """Extract text from uploaded file"""
    try:
        if file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        else:  # txt, md
            return file.getvalue().decode('utf-8')
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

def search_documents(query, documents):
    """Simple keyword search in documents"""
    if not documents:
        return ""
    
    results = []
    query_words = query.lower().split()
    
    for doc_name, doc_text in documents.items():
        doc_lower = doc_text.lower()
        if any(word in doc_lower for word in query_words):
            # Extract relevant snippet
            lines = doc_text.split('\n')
            for line in lines:
                if any(word in line.lower() for word in query_words):
                    results.append(f"**{doc_name}:** {line[:200]}")
                    if len(results) >= 3:
                        break
            if len(results) >= 3:
                break
    
    if results:
        return "**From your documents:**\n" + "\n".join(results) + "\n\n"
    return ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    provider = st.selectbox("Provider", ["gemini", "openai"])
    response_mode = st.radio("Mode", ["concise", "detailed"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3 if response_mode == "concise" else 0.7)
    
    st.subheader("📚 RAG & Documents")
    rag_enabled = st.checkbox("Enable RAG", value=True)
    
    if rag_enabled:
        uploaded_files = st.file_uploader(
            "Upload documents (TXT, MD, PDF)",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📤 Index Documents"):
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    if text:
                        st.session_state.documents[file.name] = text
                st.success(f"✅ Indexed {len(st.session_state.documents)} documents!")
        
        st.write(f"**Documents indexed:** {len(st.session_state.documents)}")
    
    st.subheader("🌐 Web Search")
    web_search = st.checkbox("Enable Web Search", value=False)
    
    st.metric("Queries", st.session_state.query_count)

# Main
st.title("🔧 AI Knowledge Companion for Engineers")
st.markdown("Ask technical questions. Get AI responses with RAG + Web Search.")

# Chat display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Build context from documents
                doc_context = ""
                if rag_enabled and st.session_state.documents:
                    doc_context = search_documents(user_input, st.session_state.documents)
                
                if provider == "gemini":
                    api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
                    if not api_key:
                        st.error("❌ No Gemini API key in secrets")
                        st.stop()
                    
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-pro")
                    
                    system_msg = "You are a helpful engineering assistant. Be concise and practical." if response_mode == "concise" else "You are a helpful engineering assistant. Provide detailed, thorough explanations with examples."
                    
                    prompt = f"{system_msg}\n\n{doc_context}User Question: {user_input}"
                    
                    response = model.generate_content(prompt)
                    answer = response.text
                
                elif provider == "openai":
                    api_key = st.secrets.get("OPENAI_API_KEY")
                    if not api_key:
                        st.error("❌ No OpenAI API key in secrets")
                        st.stop()
                    
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    
                    system_msg = "You are a helpful engineering assistant. Be concise and practical." if response_mode == "concise" else "You are a helpful engineering assistant. Provide detailed, thorough explanations with examples."
                    
                    if doc_context:
                        system_msg += f"\n\n{doc_context}"
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        *st.session_state.chat_history[-10:]
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=temperature
                    )
                    
                    answer = response.choices[0].message.content
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.query_count += 1
                
                with st.expander("📊 Details"):
                    st.json({
                        "provider": provider,
                        "mode": response_mode,
                        "rag_docs": len(st.session_state.documents),
                        "temperature": temperature
                    })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}")

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📊 Status")
    st.write(f"Queries: {st.session_state.query_count}")
with col2:
    st.caption("📚 Documents")
    st.write(f"Indexed: {len(st.session_state.documents)}")
with col3:
    st.caption("⚙️ Provider")
    st.write(provider.upper())