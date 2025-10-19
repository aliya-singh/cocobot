import streamlit as st
import requests
import json

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
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3 if response_mode == "concise" else 0.7)
    
    st.subheader("📚 RAG")
    rag_enabled = st.checkbox("Enable RAG", value=True)
    
    if rag_enabled:
        uploaded_files = st.file_uploader("Upload (TXT, MD)", type=['txt', 'md'], accept_multiple_files=True)
        if uploaded_files and st.button("Index"):
            for file in uploaded_files:
                try:
                    text = file.getvalue().decode('utf-8')
                    st.session_state.documents[file.name] = text[:2000]
                except:
                    pass
            st.success(f"✅ Indexed {len(st.session_state.documents)} docs")
        st.write(f"Docs: {len(st.session_state.documents)}")
    
    st.metric("Queries", st.session_state.query_count)

# Main
st.title("🔧 AI Knowledge Companion")
st.write("Ask technical questions.")

# Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
                if not api_key:
                    st.error("❌ No Groq API key in secrets")
                    st.stop()
                
                # Build context
                context = ""
                if rag_enabled and st.session_state.documents:
                    context = "DOCUMENTS:\n"
                    for doc_name, doc_text in st.session_state.documents.items():
                        context += f"{doc_name}:\n{doc_text}\n---\n"
                
                system = "Be concise." if response_mode == "concise" else "Be detailed and thorough."
                
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
                else:
                    answer = f"❌ Error {response.status_code}: {response.text}"
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.query_count += 1
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption(f"Queries: {st.session_state.query_count} | Docs: {len(st.session_state.documents)}")