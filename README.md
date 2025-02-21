---
title: Fastener Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Documentation for RAG System with Image Processing

## Overview
This system combines **Retrieval-Augmented Generation (RAG)** with **image classification** to recognize geometric shapes and recommend the right fasteners and manufacturing options. Users upload an image, the system analyzes it, and a chatbot provides relevant suggestions.

# Image + RAG System for Fastener Selection

This system helps users identify geometric shapes and find the best fasteners using **RAG** and **image classification**. Users upload an image, the system processes it, and a chatbot retrieves the most relevant information.

## System Components

### 1. Image Processing and Classification
The system first recognizes the geometry of the uploaded image, as different shapes require different fasteners. We use **ResNet50** from `torchvision.models`, a deep learning model for image classification, to extract key features from images. These features, called **embeddings**, allow the system to compare shapes.

- The system creates embeddings for reference shapes (e.g., flat, cylindrical, and complex geometries) using **ResNet50** and **Torch**.
- These embeddings are stored as vectors using `NumPy` for fast and efficient retrieval.
- When a user uploads an image, the system generates an embedding and compares it to the reference embeddings using **cosine similarity** (`sklearn.metrics.pairwise`).
- A similarity score close to **1** means a strong match, while a lower score suggests a different shape.

To improve accuracy, we use a **Cross-Encoder reranker** (`sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`), which directly compares query-document pairs to refine the ranking.

### 2. Document Processing & Retrieval
Once the system identifies the shape, it retrieves relevant fastener recommendations from **.docx documents** containing manufacturing guidelines.

- **Extracting Text:** The system processes documents using `python-docx` and breaks the text into **1500-character** chunks for easier retrieval.
- **Creating Embeddings:** Text chunks are converted into embeddings using **`HuggingFaceEmbeddings` (`BAAI/bge-base-en-v1.5`)**.
- **Vector Search:** These embeddings are stored in **FAISS (Facebook AI Similarity Search)** for fast lookups.
- **Matching Queries:** When a user asks a question, the system creates an embedding of the query and finds the most relevant text chunks using **cosine similarity** in FAISS.
- Only chunks with a similarity score above **0.5** are considered relevant.

To further refine results, we use **semantic filtering** and **Cross-Encoder reranking** to ensure only the most useful information is retrieved.

### 3. Query Validation & Response Generation
Finding relevant text isn’t enough—we need to make sure it actually answers the user’s question.

- **LLM Response Generation:** We use **Falcon-40B-Instruct** via `HuggingFaceEndpoint` to generate answers.
- **Checking Relevance:** The system checks if the retrieved text matches the question using `sentence-transformers`.
- **Fallback Handling:** If the system finds no useful information, it ignores irrelevant text and gives a fallback response.

This makes sure the chatbot provides accurate and meaningful answers, not just random AI-generated text.  

The **LLM uses both the user’s question and retrieved documents** to generate a final response. This way, the answer is based on real information instead of making things up.

### 4. Chatbot & User Interaction
Users talk to the system through a **Gradio chatbot** (`gradio.Blocks`). The chatbot helps users:

1. **Upload an Image:** The system detects its shape and finds relevant fastener details.
2. **Ask Questions:** Users can ask about fasteners and manufacturing based on the detected shape.
3. **Get Answers:** The **RAG model** fetches relevant documents and the LLM generates a response.

The chatbot updates responses as users interact, making sure the answers stay relevant. **Gradio** makes it easy to connect different models and create a simple, user-friendly chat interface.

### Summary
- **Image Processing**: Uses `torchvision.models` (ResNet50) to classify geometry and compare embeddings using `sklearn` cosine similarity.
- **RAG Retrieval**: Uses `sentence-transformers` to convert text into embeddings and FAISS for fast vector retrieval.
- **Query Validation**: Ensures high-quality responses by filtering irrelevant document chunks before response generation using `spacy` and `nltk`.
- **LLM Response**: Uses `transformers` to generate responses from `Mistral-7B-Instruct-v0.2`.
- **User Interaction**: Provides an interactive chatbot interface using `Gradio` for easy image uploads and queries.

This documentation explains how the system works step by step, ensuring an intuitive and efficient user experience.
