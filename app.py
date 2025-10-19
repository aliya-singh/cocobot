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
        uploaded_files = st.file_uploader("Upload (TXT, MD, PDF)", type=['txt', 'md', 'pdf'], accept_multiple_files=True)
        if uploaded_files and st.button("Index"):
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        import PyPDF2
                        reader = PyPDF2.PdfReader(file)
                        text = "".join([page.extract_text() for page in reader.pages])
                    else:
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
                api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
                if not api_key:
                    st.error("❌ No API key in secrets")
                    st.stop()
                
                # Build context
                context = ""
                if rag_enabled and st.session_state.documents:
                    context = "From your documents:\n"
                    for doc_name, doc_text in st.session_state.documents.items():
                        if any(word in doc_text.lower() for word in user_input.lower().split()):
                            context += f"\n{doc_name}: {doc_text[:500]}\n"
                
                system = "Be concise." if response_mode == "concise" else "Be detailed and thorough."
                
                # Direct Google API call
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{system}\n\n{context}\n\nUser: {user_input}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": 2000
                    }
                }
                
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['candidates'][0]['content']['parts'][0]['text']
                else:
                    answer = f"API Error: {response.status_code}\n{response.text}"
                
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.query_count += 1
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption(f"Queries: {st.session_state.query_count} | Docs: {len(st.session_state.documents)}")