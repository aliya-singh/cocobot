import streamlit as st
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Companion", page_icon="⚙️", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    provider = st.selectbox("Provider", ["gemini", "openai"])
    response_mode = st.radio("Mode", ["concise", "detailed"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3 if response_mode == "concise" else 0.7)
    
    st.metric("Queries", st.session_state.query_count)

# Main
st.title("🔧 AI Knowledge Companion")
st.write("Ask technical questions. Get AI responses.")

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
        with st.spinner("Thinking..."):
            try:
                if provider == "gemini":
                    api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
                    if not api_key:
                        st.error("❌ No Gemini key in secrets")
                        st.stop()
                    
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    system_msg = "You are a helpful engineering assistant. Be concise." if response_mode == "concise" else "You are a helpful engineering assistant. Be detailed and thorough."
                    
                    prompt = f"{system_msg}\n\nUser: {user_input}"
                    response = model.generate_content(prompt)
                    answer = response.text
                
                elif provider == "openai":
                    api_key = st.secrets.get("OPENAI_API_KEY")
                    if not api_key:
                        st.error("❌ No OpenAI key in secrets")
                        st.stop()
                    
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    
                    system_msg = "You are a helpful engineering assistant. Be concise." if response_mode == "concise" else "You are a helpful engineering assistant. Be detailed and thorough."
                    
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
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}")

st.divider()
st.caption(f"Provider: {provider} | Queries: {st.session_state.query_count}")