# Gemini-Powered PDF Chat Assistant

Upload PDFs and chat with your documents using Google Gemini AI. This project enables users to interact with their own PDF documents through a conversational AI interface, leveraging semantic search, Retrieval-Augmented Generation (RAG), and document chunking for precise, context-aware answers.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Gemini-Powered PDF Chat Assistant** is a full-stack application that allows users to upload PDFs and chat with their documents using Google's Gemini AI. The system processes PDFs, splits them into semantic chunks, stores them in MongoDB, and uses Gemini for both general chat and document-based Q&A. The backend is built with FastAPI, and the frontend uses Streamlit for an interactive user experience[^1][^2][^3][^4].

---

## Features

- Upload and process multiple PDF files per user
- Semantic chunking and storage of PDF content
- Gemini-powered Retrieval-Augmented Generation (RAG) for document Q&A
- General conversational AI when no documents are present
- Chat history and document statistics per user
- Redis caching for fast access to recent conversation history
- REST API for integration with other apps or frontends
- Streamlit-based frontend for easy interaction[^2][^1][^4]

---

## Architecture

- **Frontend:** Streamlit app for file upload and chat interface[^1]
- **Backend:** FastAPI application exposing endpoints for upload, chat, history, and document management[^2][^5]
- **Document Processing:** LangChain for PDF parsing and chunking; Google Gemini for embedding and generation[^2][^4]
- **Database:** MongoDB for persistent storage of messages and document chunks[^2][^5]
- **Cache:** Redis for fast retrieval of recent chat history[^2][^5][^6]

---

## Requirements

- Python 3.9+
- FastAPI
- Streamlit
- pymongo
- redis-py
- python-dotenv
- google-generativeai
- langchain, langchain_community
- PyPDF2, PyPDFLoader
- (Optional) Uvicorn for development server

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/gemini-pdf-chat.git
   cd gemini-pdf-chat
