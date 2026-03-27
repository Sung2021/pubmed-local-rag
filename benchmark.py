"""
benchmark.py — Compare embedding models and LLMs on the same questions.

Usage:
    python benchmark.py

Results are saved to benchmark_results.csv
"""

import os
import re
import time
import csv
from Bio import Entrez
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configuration ─────────────────────────────────────────────────────────────

ENTREZ_EMAIL = "your@email.com"
SEARCH_TERM  = "single cell RNA-seq cancer"
RETMAX       = 10

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",       # fast, small
    "all-mpnet-base-v2",       # more accurate, slower
]

LLM_MODELS = [
    "llama3.2",
    "mistral",
]

QUESTIONS = [
    "What is single cell RNA-seq?",
    "How is scRNA-seq used in cancer research?",
    "What cell types were identified in the studies?",
    "What are the key findings about tumor microenvironment?",
    "What methods were used for data analysis?",
]

# ── Fetch abstracts (once) ────────────────────────────────────────────────────

print(f"Fetching abstracts for: '{SEARCH_TERM}'...")
Entrez.email = ENTREZ_EMAIL

try:
    search_handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=RETMAX)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]

    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
    abstracts = fetch_handle.read()
    fetch_handle.close()
except Exception as e:
    print(f"Failed to fetch abstracts: {e}")
    exit(1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(abstracts)
print(f"Total chunks: {len(chunks)}\n")

# ── Run benchmark ─────────────────────────────────────────────────────────────

results = []

for emb_model in EMBEDDING_MODELS:
    print(f"\n=== Embedding: {emb_model} ===")
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', emb_model)
    persist_dir = f"./chroma_db/benchmark_{safe_name}"

    print(f"  Building vector DB...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=emb_model)
        vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=persist_dir)
    except Exception as e:
        print(f"  Failed to build DB: {e}")
        continue

    retriever = vectorstore.as_retriever()

    for llm_model in LLM_MODELS:
        print(f"\n  LLM: {llm_model}")
        try:
            llm = OllamaLLM(model=llm_model)
        except Exception as e:
            print(f"  Failed to load LLM: {e}")
            continue

        prompt = PromptTemplate.from_template(
            "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        for question in QUESTIONS:
            print(f"    Q: {question[:60]}...")
            start = time.time()
            try:
                response = chain.invoke(question)
                elapsed = time.time() - start
                print(f"       ({elapsed:.1f}s)")
            except Exception as e:
                response = f"ERROR: {e}"
                elapsed = -1
                print(f"       Error: {e}")

            results.append({
                "embedding_model": emb_model,
                "llm_model": llm_model,
                "question": question,
                "response": response[:300],
                "response_time_s": round(elapsed, 2),
            })

# ── Save results ──────────────────────────────────────────────────────────────

output_file = "benchmark_results.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["embedding_model", "llm_model", "question", "response", "response_time_s"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmark complete. Results saved to {output_file}")
print(f"Total combinations tested: {len(EMBEDDING_MODELS)} embeddings × {len(LLM_MODELS)} LLMs × {len(QUESTIONS)} questions = {len(results)} runs")
