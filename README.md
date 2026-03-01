# 🤖 RAG-Powered Chatbot

A secure, production-ready chatbot using Retrieval-Augmented Generation (RAG) to answer questions from your documents.

## Features

- 🔐 Password Protection
- ⏱️ Rate Limiting (3 seconds between queries)
- 📊 Usage Tracking
- 📚 Document RAG with PDF support
- 💬 Conversational Memory
- 🎯 Source Attribution

## Deployment on Streamlit Cloud

1. Push to GitHub (public repository)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secret: `OPENAI_API_KEY = "your-key"`
5. Deploy!

## Security

Generate new hash:
```python
import hashlib
password = "your_secure_password"
print(hashlib.sha256(password.encode()).hexdigest())
```

Update the hash in `app.py` (line ~35).
# rag-chatbot
