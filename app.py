import streamlit as st
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🔧 AI Knowledge Companion",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INITIALIZE SESSION ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider selection
    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Select Provider",
        ["openai", "gemini"],
        help="OpenAI: gpt-3.5-turbo | Gemini: gemini-1.5-flash"
    )
    
    # Response mode
    st.subheader("Response Style")
    response_mode = st.radio("Mode", ["concise", "detailed"])
    
    # Temperature
    temp_default = 0.3 if response_mode == "concise" else 0.7
    temperature = st.slider("Temperature", 0.0, 1.0, temp_default, 0.1)
    
    # RAG Section
    st.subheader("📚 RAG & Documents")
    st.toggle("Enable RAG", value=True, key="rag_toggle")
    st.session_state.rag_enabled = st.session_state.rag_toggle
    
    if st.session_state.rag_enabled:
        uploaded_files = st.file_uploader(
            "Upload documents (TXT, MD, PDF)",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"✅ {len(uploaded_files)} file(s) ready")
            if st.button("📤 Index Documents"):
                st.session_state.uploaded_docs = uploaded_files
                st.success(f"Indexed {len(uploaded_files)} documents!")
        
        st.write(f"**Documents indexed:** {len(st.session_state.uploaded_docs)}")
    
    # Web Search
    st.subheader("🌐 Web Search")
    enable_web_search = st.toggle("Enable Web Search", value=False)
    
    # Stats
    st.subheader("📊 Statistics")
    st.metric("Queries", st.session_state.query_count)
    st.metric("Documents", len(st.session_state.uploaded_docs))

# ==================== MAIN CHAT INTERFACE ====================
st.title("🔧 AI Knowledge Companion for Engineers")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.markdown("""
    **Welcome!** Ask technical questions and get intelligent responses.
    
    - 💬 Chat with AI
    - 📚 Search your documents (RAG)
    - 🌐 Get real-time web answers
    - ⚡ Choose response depth
    """)

with col2:
    st.markdown(f"""
    **Provider:** {provider.upper()}
    
    **Mode:** {response_mode}
    
    **Temp:** {temperature}
    
    **RAG:** {'✅' if st.session_state.rag_enabled else '❌'}
    """)

# Display chat history
st.subheader("💬 Conversation")

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                st.caption(f"📚 {message['source']}")

# ==================== HELPER FUNCTIONS ====================

def get_system_prompt(mode: str, has_docs: bool = False) -> str:
    """Get appropriate system prompt"""
    if mode == "concise":
        return """You are an AI Knowledge Companion for Engineers.
Provide brief, focused answers (2-3 sentences max).
Be direct and practical."""
    else:
        base = """You are an AI Knowledge Companion for Engineers.
Provide comprehensive, detailed explanations with examples.
Discuss trade-offs and best practices."""
        if has_docs:
            base += "\nReference the provided documents when available."
        return base

def search_documents(query: str, docs_content: list) -> str:
    """Simple document search"""
    if not docs_content:
        return ""
    
    # Very simple keyword matching
    relevant = []
    query_lower = query.lower()
    
    for doc in docs_content:
        if any(word in doc.lower() for word in query_lower.split()):
            relevant.append(doc[:500])
    
    if relevant:
        return "**From your documents:**\n" + "\n---\n".join(relevant)
    return ""

def search_web(query: str) -> str:
    """Simple web search using DuckDuckGo"""
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {'q': query, 'format': 'json'}
        response = requests.get('https://api.duckduckgo.com', params=params, headers=headers, timeout=5)
        data = response.json()
        
        results = []
        if data.get('AbstractText'):
            results.append(f"**Direct Answer:** {data['AbstractText']}")
        
        for result in data.get('Results', [])[:3]:
            if result.get('Text'):
                results.append(f"- {result['Text']}")
        
        if results:
            return "**Web Search Results:**\n" + "\n".join(results)
    except:
        pass
    
    return ""

# ==================== INPUT & PROCESSING ====================
user_input = st.chat_input("Ask me anything about engineering, code, DevOps...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get API key
                if provider == "openai":
                    api_key = st.secrets.get("OPENAI_API_KEY")
                    if not api_key:
                        st.error("❌ OpenAI API key not in secrets")
                        st.stop()
                    
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    
                    # Build context
                    context = ""
                    
                    # RAG search
                    if st.session_state.rag_enabled and st.session_state.uploaded_docs:
                        doc_contents = []
                        for doc in st.session_state.uploaded_docs:
                            try:
                                if doc.type == 'application/pdf':
                                    import PyPDF2
                                    reader = PyPDF2.PdfReader(doc)
                                    text = "".join([page.extract_text() for page in reader.pages])
                                else:
                                    text = doc.getvalue().decode()
                                doc_contents.append(text[:1000])
                            except:
                                pass
                        
                        doc_context = search_documents(user_input, doc_contents)
                        if doc_context:
                            context += doc_context + "\n\n"
                    
                    # Web search
                    if enable_web_search:
                        web_context = search_web(user_input)
                        if web_context:
                            context += web_context + "\n\n"
                    
                    # Build messages
                    system_prompt = get_system_prompt(response_mode, bool(context))
                    if context:
                        system_prompt += f"\n\n{context}"
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        *st.session_state.chat_history[-10:]
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2000
                    )
                    
                    answer = response.choices[0].message.content
                
                elif provider == "gemini":
                    api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
                    if not api_key:
                        st.error("❌ Gemini API key not in secrets")
                        st.stop()
                    
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    # Build context
                    context = ""
                    
                    if st.session_state.rag_enabled and st.session_state.uploaded_docs:
                        doc_contents = []
                        for doc in st.session_state.uploaded_docs:
                            try:
                                if doc.type == 'application/pdf':
                                    import PyPDF2
                                    reader = PyPDF2.PdfReader(doc)
                                    text = "".join([page.extract_text() for page in reader.pages])
                                else:
                                    text = doc.getvalue().decode()
                                doc_contents.append(text[:1000])
                            except:
                                pass
                        
                        doc_context = search_documents(user_input, doc_contents)
                        if doc_context:
                            context += doc_context + "\n\n"
                    
                    if enable_web_search:
                        web_context = search_web(user_input)
                        if web_context:
                            context += web_context + "\n\n"
                    
                    system_prompt = get_system_prompt(response_mode, bool(context))
                    
                    full_prompt = system_prompt
                    if context:
                        full_prompt += f"\n\n{context}"
                    
                    full_prompt += f"\n\nUser: {user_input}"
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text
                
                st.markdown(answer)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
                
                st.session_state.query_count += 1
                
                # Show metadata
                with st.expander("📊 Response Details"):
                    st.json({
                        "provider": provider,
                        "mode": response_mode,
                        "temperature": temperature,
                        "rag_enabled": st.session_state.rag_enabled,
                        "web_search": enable_web_search,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}")

# ==================== FOOTER ====================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎯 Features")
    st.markdown("""
    - Chat with AI
    - Document RAG
    - Web search
    - Response modes
    """)

with col2:
    st.caption("📊 Status")
    st.markdown(f"""
    - Provider: {provider}
    - Queries: {st.session_state.query_count}
    - Docs: {len(st.session_state.uploaded_docs)}
    - Mode: {response_mode}
    """)

with col3:
    st.caption("🔗 Resources")
    st.markdown("""
    - [OpenAI Docs](https://platform.openai.com/docs)
    - [Gemini Docs](https://ai.google.dev/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    """)