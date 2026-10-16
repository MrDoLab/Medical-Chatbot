# MedLink — Medical RAG Chatbot

MedLink is a medical information chatbot built with LangGraph-based Retrieval-Augmented Generation (RAG). It retrieves information from multiple medical sources and uses GPT-4o to generate responses based on the retrieved content.

## Features

- **Multi-Source Retrieval:** Parallel retrieval from PubMed, Tavily, MedGemma, and Amazon Bedrock Knowledge Base.
- **RAG Pipeline:** LangGraph orchestrates question processing, retrieval, GPT-4o answer integration, and LLM-based response evaluation.
- **Web Interface:** React-based chat interface with Markdown rendering and a medical news ticker.

## Architecture

```text
React Chat Interface
        |
        v
FastAPI + LangGraph
        |
        v
Parallel Retrieval
  ├── PubMed
  ├── Tavily
  ├── MedGemma
  └── Bedrock Knowledge Base
        |
        v
GPT-4o Answer Integration
        |
        v
LLM-based Response Evaluation
        |
        v
Response Formatting / Output
```

## Tech Stack

- **Frontend:** React, Vite, Tailwind CSS
- **Backend:** Python, FastAPI, Flask
- **LLM & RAG:** GPT-4o, MedGemma, LangChain, LangGraph
- **Retrieval:** PubMed, Tavily, Amazon Bedrock Knowledge Base
