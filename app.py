import streamlit as st
import logging
from pathlib import Path
from datetime import datetime

# Import all modules
from config.config import Config
from models.llm import LLMFactory
from models.embeddings import get_embedding_model
from utils.rag import get_rag_engine
from utils.web_search import get_search_manager
from utils.prompts import PromptTemplates
from utils.rate_limiter import get_rate_limiter
from utils.token_counter import TokenCounter
from utils.cache import get_cache_manager, QueryCache

# Initialize config
Config.ensure_directories()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== STREAMLIT PAGE CONFIG ====================
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = get_rag_engine()

if "search_manager" not in st.session_state:
    st.session_state.search_manager = get_search_manager()

if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = get_rate_limiter(
        Config.MAX_QUERIES_PER_DAY,
        Config.RATE_LIMIT_SECONDS
    )

if "cache_manager" not in st.session_state:
    st.session_state.cache_manager = get_cache_manager()

if "query_cache" not in st.session_state:
    st.session_state.query_cache = QueryCache(st.session_state.cache_manager)


# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider Selection
    st.subheader("LLM Provider")
    available_providers = Config.get_available_providers()
    
    if not available_providers:
        st.error("❌ No API keys configured! Please add them to .env file")
        st.stop()
    
    selected_provider = st.selectbox(
        "Select LLM Provider",
        available_providers,
        index=0 if "groq" in available_providers else 0,
        help="Groq is free and fastest. OpenAI is most capable but costs money."
    )
    
    # Response Mode
    st.subheader("Response Style")
    response_mode = st.radio(
        "Choose response mode:",
        ["concise", "detailed"],
        help="Concise: Quick answers. Detailed: In-depth explanations"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (Creativity)",
        0.0, 1.0,
        Config.TEMP_CONCISE if response_mode == "concise" else Config.TEMP_DETAILED,
        0.1,
        help="Lower = more focused. Higher = more creative"
    )
    
    # RAG Settings
    st.subheader("📚 RAG & Documents")
    
    enable_rag = st.checkbox("Enable RAG (Use local documents)", value=True)
    
    if enable_rag:
        # Show RAG stats
        rag_stats = st.session_state.rag_engine.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", rag_stats['total_documents'])
        with col2:
            st.metric("Chunks", rag_stats['total_chunks'])
        
        # Upload documents
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload TXT, MD, or PDF files",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            help="Supported: TXT, Markdown, PDF"
        )
        
        if uploaded_files:
            if st.button("📤 Process & Index Documents"):
                with st.spinner("Processing documents..."):
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        # Save file temporarily
                        temp_path = Config.UPLOADED_DOCS_DIR / uploaded_file.name
                        temp_path.write_bytes(uploaded_file.getbuffer())
                        file_paths.append(str(temp_path))
                    
                    # Add to RAG
                    result = st.session_state.rag_engine.add_documents(file_paths)
                    st.success(f"✅ Indexed {result['chunks_added']} chunks from {result['documents_added']} documents!")
    
    # Web Search Settings
    st.subheader("🌐 Web Search")
    enable_web_search = st.checkbox("Enable Web Search", value=Config.ENABLE_WEB_SEARCH)
    
    if enable_web_search:
        search_threshold = st.slider(
            "Search Confidence Threshold",
            0.0, 1.0, Config.WEB_SEARCH_CONFIDENCE_THRESHOLD, 0.1,
            help="Search only if confidence is below this threshold"
        )
    
    # Rate Limit Info
    st.subheader("📊 Usage Statistics")
    rate_status = st.session_state.rate_limiter.get_status("default")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Queries Today", f"{rate_status['requests_today']}/{rate_status['limit']}")
    with col2:
        st.metric("Remaining", rate_status['remaining'])
    with col3:
        st.metric("Usage %", f"{rate_status['percentage_used']}%")
    
    if rate_status['reset_in_hours'] > 0:
        st.info(f"⏰ Resets in {rate_status['reset_in_hours']} hours")
    
    # Cache Info
    cache_stats = st.session_state.cache_manager.get_cache_stats()
    st.subheader("💾 Cache")
    st.caption(f"Cached entries: {cache_stats['total_entries']}")
    st.caption(f"Cache size: {cache_stats['total_size_mb']} MB")
    
    # Clear options
    if st.button("🗑️ Clear Cache"):
        st.session_state.cache_manager.clear_expired(0)
        st.success("Cache cleared!")
    
    if st.button("🗑️ Clear Documents"):
        st.session_state.rag_engine.clear()
        st.success("Documents cleared!")


# ==================== MAIN CHAT INTERFACE ====================
st.title(Config.APP_TITLE)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.markdown("""
    Welcome to your **AI Knowledge Companion for Engineers**! 
    
    - 💭 Ask technical questions
    - 📚 Get answers from your uploaded documents
    - 🌐 Get real-time web search when needed
    - ⚡ Choose response depth (concise or detailed)
    """)

with col2:
    st.markdown("""
    ### 🚀 Quick Features
    - **Groq**: Free, instant (default)
    - **RAG**: Local document search
    - **Web Search**: Real-time answers
    - **Smart Cache**: Save API costs
    """)

# Display chat history
st.subheader("💬 Conversation")

chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("📊 Metadata"):
                    st.json(message["metadata"])


# ==================== INPUT & PROCESSING ====================
user_input = st.chat_input(
    "Ask me anything about engineering, code, DevOps, architecture...",
    key="user_input"
)

if user_input:
    # Check rate limit
    rate_check = st.session_state.rate_limiter.is_allowed("default")
    
    if not rate_check["allowed"]:
        st.error(f"⏸️ {rate_check['reason']}")
        if "reset_in_seconds" in rate_check:
            st.warning(f"Try again in {rate_check['reset_in_seconds']} seconds")
        st.stop()
    
    # Add user message to history
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
                # Initialize LLM if needed
                if st.session_state.llm_instance is None or \
                   st.session_state.llm_instance.model != getattr(Config, f"{selected_provider.upper()}_MODEL", ""):
                    
                    api_key = getattr(Config, f"{selected_provider.upper()}_API_KEY")
                    st.session_state.llm_instance = LLMFactory.create_llm(
                        selected_provider,
                        api_key,
                        temperature=temperature
                    )
                
                # Check cache
                cached_response = st.session_state.query_cache.get_response(
                    user_input,
                    selected_provider,
                    Config.CACHE_TTL_HOURS
                )
                
                metadata = {
                    "provider": selected_provider,
                    "mode": response_mode,
                    "temperature": temperature,
                    "timestamp": datetime.now().isoformat()
                }
                
                if cached_response:
                    st.info("📌 From cache")
                    response = cached_response
                    metadata["cached"] = True
                else:
                    # Build prompt with optional RAG context
                    context = ""
                    rag_results = []
                    
                    if enable_rag:
                        # Search RAG
                        rag_results = st.session_state.rag_engine.search(user_input, top_k=5)
                        
                        if rag_results:
                            context = PromptTemplates.format_rag_context(rag_results)
                            metadata["rag_results"] = len(rag_results)
                        
                        # Check if we should do web search
                        confidence = st.session_state.rag_engine.get_search_confidence(user_input)
                        
                        if enable_web_search and confidence < search_threshold:
                            web_results = st.session_state.search_manager.search(user_input, max_results=3)
                            if web_results:
                                web_context = PromptTemplates.format_web_search_context(web_results)
                                context += "\n\n**Web Search Results:**\n" + web_context
                                metadata["web_search"] = True
                                metadata["web_results"] = len(web_results)
                    
                    # Build messages
                    system_prompt = PromptTemplates.build_system_prompt(response_mode, include_rag=bool(context))
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        *st.session_state.chat_history[-Config.MAX_CONVERSATION_HISTORY:],
                    ]
                    
                    if context:
                        messages[-1]["content"] = f"{messages[-1]['content']}\n\n**Context:**\n{context}"
                    
                    # Estimate tokens before calling
                    input_tokens = TokenCounter.estimate_tokens(str(messages), selected_provider)
                    output_tokens = TokenCounter.estimate_response_tokens(input_tokens, selected_provider)
                    estimated_cost = TokenCounter.estimate_cost(input_tokens, output_tokens, selected_provider)
                    
                    metadata["estimated_tokens"] = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "cost_usd": estimated_cost
                    }
                    
                    # Get response
                    response = st.session_state.llm_instance.chat(messages)
                    
                    # Cache response
                    st.session_state.query_cache.cache_response(
                        user_input,
                        response,
                        selected_provider
                    )
                
                # Display response
                st.markdown(response)
                
                # Store in history with metadata
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": metadata
                })
                
                # Show metadata expander
                with st.expander("📊 Response Details"):
                    st.json(metadata)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error generating response: {e}")


# ==================== FOOTER ====================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔧 Model Info")
    if st.session_state.llm_instance:
        model_info = st.session_state.llm_instance.get_model_info()
        st.json(model_info)

with col2:
    st.caption("📚 RAG Status")
    rag_stats = st.session_state.rag_engine.get_stats()
    st.json({
        "documents": rag_stats['total_documents'],
        "chunks": rag_stats['total_chunks'],
        "embedding_model": rag_stats['embedding_model']['model_name']
    })

with col3:
    st.caption("💡 Tips")
    st.markdown("""
    - Use **concise mode** for quick answers
    - Use **detailed mode** for learning
    - Upload docs for **instant context**
    - Web search works when needed
    - **Groq is free & fast!**
    """)