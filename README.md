# 🔧 AI Knowledge Companion for Engineers

An intelligent AI-powered chatbot that helps engineers solve technical problems using Retrieval-Augmented Generation (RAG). Upload documents, paste website content, or add text to create a personalized knowledge base and get instant AI-powered answers.

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)
![Groq](https://img.shields.io/badge/Groq-Free%20API-00D084?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## ✨ Features

- **💬 AI Chat** - Ask questions and get intelligent responses powered by Groq's Llama 3.3 70B model
- **📚 RAG System** - Upload documents (PDF, TXT, MD) and search through them semantically
- **🌐 Website Content** - Paste website URLs to fetch and index content automatically
- **📝 Text Input** - Add raw text snippets to your knowledge base
- **🔍 Smart Search** - Keyword-based document search with relevance ranking
- **⚡ Multiple Sources** - Index up to 20 documents from different sources
- **🎯 Response Modes** - Choose between concise (quick) or detailed (thorough) responses
- **🎛️ Temperature Control** - Adjust AI creativity with temperature slider
- **📊 Statistics** - Track queries, indexed sources, and chat history

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Groq API key (free at https://console.groq.com/)
- Streamlit account for deployment (optional)

### Local Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd cocobot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo GROQ_API_KEY=your_api_key_here > .env

# 5. Run the app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Click "New app" and connect your repository
4. In app settings, add this secret:
   ```
   GROQ_API_KEY = "your_api_key_here"
   ```
5. Deploy!

## 📖 Usage

### Adding Documents

**Method 1: Upload Files**
- Click "📄 Files" tab in sidebar
- Upload TXT, MD, or PDF files
- Click "📤 Index Files"

**Method 2: Add Website**
- Click "🌐 Website" tab
- Paste URL (e.g., https://example.com)
- Click "🌐 Fetch Website"
- Content is automatically extracted

**Method 3: Paste Text**
- Click "📋 Text" tab
- Paste your content
- Give it a name
- Click "📝 Add Text"

### Asking Questions

1. Type your question in the chat input
2. AI searches your knowledge base
3. Get intelligent answer with sources cited

### Customizing Responses

- **Mode**: Choose "concise" for quick answers or "detailed" for thorough explanations
- **Temperature**: Adjust creativity (0.0 = focused, 1.0 = creative)

## 🏗️ Project Structure

```
cocobot/
├── config/
│   └── config.py              # Configuration settings
├── models/
│   └── groq_client.py         # Groq API client
├── utils/
│   ├── document_handler.py    # File/website extraction
│   └── search.py              # Document search logic
├── app.py                     # Main Streamlit app
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python runtime version
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔧 Configuration

Edit `config/config.py` to customize:

```python
# API Settings
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama-3.3-70b-versatile"

# RAG Settings
MAX_DOCUMENT_SIZE = 3000      # Max characters per document
MAX_DOCUMENTS = 20             # Max indexed documents
MAX_TOKENS = 2000              # Max response tokens

# Response Settings
DEFAULT_TEMPERATURE_CONCISE = 0.3
DEFAULT_TEMPERATURE_DETAILED = 0.7
```

## 📋 Requirements

- `streamlit==1.28.1` - Web UI framework
- `requests==2.31.0` - HTTP requests
- `PyPDF2==3.0.1` - PDF text extraction
- `beautifulsoup4==4.12.0` - Website content extraction

## 🆓 Completely Free!

- **Groq API**: Unlimited free tier (no credit card needed after verification)
- **Streamlit Cloud**: Free tier for deployment
- **No subscription required**

## 📊 How It Works

```
User Input
    ↓
Document Search (RAG)
    ↓
Build Context
    ↓
Send to Groq API
    ↓
Stream Response
    ↓
Display Answer with Sources
```

## 🎯 Use Cases

- **Engineering Documentation** - Index company docs and get instant answers
- **Learning** - Upload tutorials and ask questions about concepts
- **Technical Research** - Add websites and papers to knowledge base
- **Code Reference** - Store code snippets and coding guidelines
- **Meeting Notes** - Index meeting notes and search for decisions
- **API Documentation** - Add API docs and ask how to use them

## ⚙️ API Endpoints

### Groq API
- **Model**: `llama-3.3-70b-versatile`
- **Endpoint**: `https://api.groq.com/openai/v1/chat/completions`
- **Rate Limit**: Unlimited on free tier
- **Response Time**: ~1-2 seconds

## 🐛 Troubleshooting

### "No API key in secrets"
- Add `GROQ_API_KEY` to Streamlit Cloud Secrets
- Or create `.env` file locally with your key

### "PDF extraction error"
- Ensure PDF is text-based (not scanned image)
- Try uploading smaller PDFs

### "Website fetch error"
- Check URL is valid and accessible
- Website might have restrictions

### "No documents indexed"
- Upload at least one document
- Check file format (TXT, MD, PDF)

## 📝 Example Queries

```
"What is Docker and how do I use it?"
"Explain the difference between REST and GraphQL"
"How do I fix a memory leak in Python?"
"Summarize the security best practices from this document"
"What are the key points from my uploaded file?"
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Push and create a pull request

## 📄 License

MIT License - feel free to use this project however you want!

## 🙋 Support

- **Questions?** Check the troubleshooting section
- **Bug?** Open an issue on GitHub
- **Ideas?** Start a discussion

## 🎓 Learning Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [Groq API Docs](https://console.groq.com/docs)
- [RAG Explained](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [BeautifulSoup Docs](https://www.crummy.com/software/BeautifulSoup/)

## 🚀 What's Next?

- [ ] Add conversation memory across sessions
- [ ] Support for more document formats (DOCX, PPT)
- [ ] Vector database integration for better search
- [ ] Multi-user support with user authentication
- [ ] Export chat history
- [ ] Custom prompt templates
- [ ] Document metadata and tagging

## 💡 Tips & Tricks

- **Better Answers**: Upload relevant documentation first
- **Fast Responses**: Use concise mode for quick answers
- **Accurate Info**: Keep temperature low (0.3) for factual questions
- **Creative Ideas**: Increase temperature (0.7) for brainstorming
- **Multiple Sources**: Combine different document types for better context

## 📞 Contact

- GitHub: [Your GitHub]
- Email: [Your Email]
- Twitter: [Your Twitter]

---

**Made with ❤️ for Engineers**

Built with Streamlit, Groq, and BeautifulSoup

⭐ If you find this helpful, please star the repository!