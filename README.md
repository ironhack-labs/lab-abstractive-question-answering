# Abstractive Question Answering with RAG

An end-to-end pipeline that generates natural language answers to open-ended questions by retrieving semantically relevant Wikipedia passages and synthesising a response. → [Notebook](lab_abstractive_question_answering(1).ipynb)

## What it demonstrates

This is a working implementation of **Retrieval-Augmented Generation (RAG)** — the architecture that sits behind most production Q&A systems today. Instead of a model trying to answer from memory alone, it first retrieves relevant context, then generates an answer grounded in that context. The notebook makes the failure mode explicit too: when no relevant context is found (e.g. a question about COVID-19 against a history-only dataset), the generator hallucinates — showing why retrieval quality is the bottleneck, not generation.

## Key technical decisions worth noting

- **Semantic search, not keyword search.** Passages are embedded as 768-dimensional vectors (MPNet SentenceTransformer) and stored in Pinecone. A query for "Napoleon's defeat" surfaces documents about "Bonaparte's final campaign" or "the Battle of Waterloo's outcome" — exact phrase matching would miss these entirely.
- **Cosine similarity as the distance metric.** The retriever (microsoft/mpnet-base) is explicitly optimised for cosine similarity, so the Pinecone index is configured to match. Using dot product or Euclidean distance here would degrade retrieval quality.
- **Streaming dataset loading.** The source dataset (Wiki Snippets) is 9 GB. The notebook loads it in streaming mode so only the 10,000 filtered "History" passages are ever pulled into memory.
- **ELI5-BART as the generator.** This seq2seq model was fine-tuned on the "Explain Like I'm 5" dataset, making it better at synthesising readable answers from noisy retrieved contexts rather than copying spans verbatim (extractive QA).
- **Batched upsert.** Embeddings are generated and pushed to Pinecone in batches to avoid memory pressure and reduce API round-trips.

## Stack

| Component | Tool |
|---|---|
| Vector database | Pinecone |
| Retriever | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| Generator | `yjernite/bart_eli5` (ELI5 BART) |
| Dataset | Wiki Snippets via HuggingFace Datasets (streaming) |
| Runtime | Python 3, Jupyter Notebook |

## How to run

1. Get a free Pinecone API key at [app.pinecone.io](https://app.pinecone.io/) and add it to `.env`:
   ```
   PINECONE_API_KEY=your_key_here
   ```
2. Install dependencies (the notebook's first cells handle this, but expect a fresh install to take a few minutes due to `transformers` + `sentence-transformers`).
3. Run the notebook top to bottom. Embedding 10,000 passages takes ~5–10 minutes on CPU; use a GPU runtime if available.
