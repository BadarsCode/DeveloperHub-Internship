# Chatbot

This folder contains a standalone chatbot implementation built from the logic in `notebook/document.ipynb`.

## What it does

- Loads PDFs recursively from `data/`
- Splits documents into chunks with `RecursiveCharacterTextSplitter`
- Generates embeddings with `SentenceTransformer`
- Stores vectors in ChromaDB
- Retrieves relevant chunks per question
- Sends context + question to Gemini for final answer
- Uses conversation memory so follow-up questions are context-aware

## Run

```powershell
python -m chatbot.main --provider gemini
```

Ask one question without interactive mode:

```powershell
python -m chatbot.main --provider gemini --question "What is PCA?"
```

Rebuild vector store:

```powershell
python -m chatbot.main --provider gemini --rebuild
```

Pass API key directly (instead of `.env`) and request longer output:

```powershell
python -m chatbot.main --provider gemini --api-key "YOUR_KEY" --max-tokens 4096
```

## Streamlit App

The Streamlit app can:

- Upload `.pdf` and `.txt` files
- Load Wikipedia topics
- Build an in-app vector store
- Chat with context-aware conversation memory
- Uses Gemini API key from `.env`
- Increase max response tokens for detailed answers

Run:

```powershell
streamlit run chatbot/streamlit_app.py
```

## Create Gemini API Key

1. Open Google AI Studio: `https://aistudio.google.com/app/apikey`
2. Sign in and click `Create API key`
3. Copy the key
4. In project root `.env`, add:
   `GEMINI_API_KEY="your_api_key_here"`
5. Restart your app
