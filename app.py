import streamlit as st
import logging
from pathlib import Path
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🔧 AI Knowledge Companion",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD CONFIG ====================
try:
    from config.config import Config
    Config.ensure_directories()
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()

# ==================== LOAD MODULES ====================
try:
    from models.llm import LLMFactory
    from utils.prompts import PromptTemplates
    from utils.token_counter import TokenCounter
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = None

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get available providers
    try:
        available_providers = Config.get_available_providers()
    except:
        available_providers = []
    
    if not available_providers:
        st.error("❌ No API keys configured!")
        st.info("Add API keys to Streamlit Cloud Secrets or .env file")
        st.stop()
    
    # Provider Selection
    st.subheader("LLM Provider")
    selected_provider = st.selectbox(
        "Select LLM Provider",
        available_providers,
        index=0 if "groq" in available_providers else 0,
        help="Groq is free and fastest"
    )
    
    # Response Mode
    st.subheader("Response Style")
    response_mode = st.radio(
        "Choose response mode:",
        ["concise", "detailed"],
        help="Concise: Quick answers. Detailed: In-depth explanations"
    )
    
    # Temperature slider
    temp_default = 0.3 if response_mode == "concise" else 0.7
    temperature = st.slider(
        "Temperature (Creativity)",
        0.0, 1.0, temp_default, 0.1,
        help="Lower = focused. Higher = creative"
    )
    
    # Stats
    st.subheader("📊 Statistics")
    st.metric("Queries This Session", st.session_state.query_count)


# ==================== MAIN CHAT INTERFACE ====================
st.title("🔧 AI Knowledge Companion for Engineers")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.markdown("""
    **Welcome!** Ask technical questions and get instant AI responses.
    
    - 💭 Chat with AI
    - 🚀 Powered by Groq (FREE!)
    - ⚡ Fast responses
    """)

with col2:
    st.markdown(f"""
    **Active Provider**: {selected_provider.upper()}
    
    **Response Mode**: {response_mode}
    
    **Temperature**: {temperature}
    """)

# Display chat history
st.subheader("💬 Conversation")

chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ==================== INPUT & PROCESSING ====================
user_input = st.chat_input(
    "Ask me anything about engineering, code, DevOps...",
    key="user_input"
)

if user_input:
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
                if st.session_state.llm_instance is None:
                    api_key_attr = f"{selected_provider.upper()}_API_KEY"
                    api_key = getattr(Config, api_key_attr, None)
                    
                    if not api_key:
                        # Try to get from Streamlit secrets
                        try:
                            api_key = st.secrets.get(api_key_attr)
                        except:
                            pass
                    
                    if not api_key:
                        st.error(f"❌ {selected_provider.upper()} API key not found")
                        st.stop()
                    
                    st.session_state.llm_instance = FLMFactory.create_llm(
                        selected_provider,
                        api_key,
                        temperature=temperature
                    )
                
                # Build prompt
                system_prompt = PromptTemplates.build_system_prompt(response_mode, include_rag=False)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    *st.session_state.chat_history[-10:],
                ]
                
                # Get response
                response = st.session_state.llm_instance.chat(messages)
                
                # Display response
                st.markdown(response)
                
                # Store in history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Update counter
                st.session_state.query_count += 1
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}")


# ==================== FOOTER ====================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎯 Quick Tips")
    st.markdown("""
    - Use **concise mode** for quick answers
    - Use **detailed mode** for learning
    - Adjust temperature for creativity
    """)

with col2:
    st.caption("🔧 About")
    st.markdown(f"""
    - **Provider**: {selected_provider}
    - **Mode**: {response_mode}
    - **Session**: {st.session_state.query_count} queries
    """)

with col3:
    st.caption("📚 Resources")
    st.markdown("""
    - [Groq API](https://console.groq.com/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    - [GitHub](https://github.com/)
    """)