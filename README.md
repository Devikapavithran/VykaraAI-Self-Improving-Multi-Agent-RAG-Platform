# Vykara – Multi-Agent AI Financial Risk Analysis System

## Overview

Vykara is an AI-powered financial intelligence system that analyzes company risks using Retrieval-Augmented Generation (RAG), hybrid document retrieval, and large language models.

The system retrieves financial context from indexed knowledge sources and generates structured insights using LLM reasoning.

It is designed as a multi-agent AI pipeline where each component performs a specialized task including query rewriting, hybrid search, reranking, and evaluation.

---

# Live Deployment

Vykara API is publicly deployed on Google Cloud Run.

Base URL

https://vykara-api-535133902430.europe-west1.run.app/docs

---

# How to Run / Test Vykara

The easiest way to test the system is using the Swagger API interface.

Open the following URL:

https://vykara-api-535133902430.europe-west1.run.app/docs

Follow the steps below.

## Step 1 – Find the Query Endpoint

Inside the Swagger interface locate:

POST /query

Click to expand the endpoint.

---

## Step 2 – Enable Request Editing

Click the **Try it out** button.

This will allow you to modify the request.

---

## Step 3 – Enter the API Key

Under **Headers**, enter:

x-api-key : vykara2026 (IMPORTANT)

---

## Step 4 – Enter a Query

Edit the request body.

Example:

{
"query": "compare amazon and nvidia risks"
}

You can replace the text with any financial risk question.

---

## Step 5 – Execute

Click **Execute**.

Vykara will perform:

query rewriting → hybrid retrieval → reranking → LLM analysis

and return a structured response.

---

# Some Sample Questions to Try

Compare Amazon and Nvidia risks

Analyze Tesla's financial risk factors

Compare Nvidia and AMD investment risks

What operational risks affect Amazon?

What long-term risks affect Tesla's EV strategy?

Compare Apple and Microsoft financial stability

What risks affect Nvidia's AI hardware dominance?

Analyze Amazon's regulatory challenges

Compare Tesla and Nvidia growth risks

What supply chain risks affect Nvidia?

---

# Why Vykara

Financial risk analysis often requires manual research across multiple sources.

Vykara automates this process by combining:

• Retrieval-Augmented Generation
• Vector similarity search
• Reranking models
• LLM reasoning

This allows faster and more structured financial insights.

---

# AI System Design

Vykara is built using a modular multi-agent pipeline.

Query Agent
Rewrites user queries for improved retrieval accuracy.

Hybrid Retrieval
Performs both vector similarity search and keyword search.

Reranking Model
Ranks results using a cross-encoder model.

LLM Reasoning
Generates financial risk analysis.

Evaluation Agent
Ensures response quality.

---

# Tech Stack

Python
FastAPI
LangChain
Google Gemini 2.0 Flash
FAISS Vector Database
Sentence Transformer Embeddings
MS MARCO Cross Encoder
Docker
Google Cloud Run

---

# Architecture

User Query
↓
Query Rewriting Agent
↓
Hybrid Retrieval Engine
↓
Reranking Model
↓
Context Aggregation
↓
Gemini LLM Analysis
↓
Evaluation Agent
↓
Final Risk Insight

---

# Author

Devika S
BTech CSE 
Lovely Professional University

---

# Vision

Vykara demonstrates how modern AI systems combine retrieval techniques, language models, and scalable cloud infrastructure to build intelligent financial analysis assistants.
