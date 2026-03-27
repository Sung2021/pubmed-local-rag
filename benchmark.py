"""
benchmark.py — Compare embedding models and LLMs on the same questions.

Usage:
    python benchmark.py

Results are saved to benchmark_results.csv

Configuration via .env:
    ENTREZ_EMAIL   your PubMed email
    SEARCH_TERM    PubMed query (default: single cell RNA-seq cancer)
    RETMAX         number of papers to fetch (default: 10)
"""

import os
import re
import time
import csv
from Bio import Entrez
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ── Configuration (override via .env) ────────────────────────────────────────

Entrez.email = os.getenv("ENTREZ_EMAIL", "your@email.com")
SEARCH_TERM  = os.getenv("SEARCH_TERM", "single cell RNA-seq cancer")
RETMAX       = int(os.getenv("RETMAX", "10"))

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",    # fast, small
    "all-mpnet-base-v2",   # more accurate, slower
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

# ── Fetch abstracts as Document objects with metadata (once) ─────────────────

print(f"Fetching abstracts for: '{SEARCH_TERM}'...")

try:
    search_handle = Entrez.esearch(db="pubmed", term=SEARCH_TERM, retmax=RETMAX)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]

    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
    records = Entrez.read(fetch_handle)
    fetch_handle.close()
except Exception as e:
    print(f"Failed to fetch abstracts: {e}")
    exit(1)

def extract_text(field):
    if isinstance(field, list):
        return " ".join(str(x) for x in field)
    return str(field) if field else ""

documents = []
for record in records["PubmedArticle"]:
    try:
        article = record["MedlineCitation"]["Article"]
        pmid    = str(record["MedlineCitation"]["PMID"])
        title   = extract_text(article.get("ArticleTitle", ""))
        journal = extract_text(article.get("Journal", {}).get("Title", ""))
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year    = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))
        abstract_texts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = extract_text(abstract_texts)
        if not abstract:
            continue
        documents.append(Document(
            page_content=abstract,
            metadata={"pmid": pmid, "title": title, "journal": journal, "year": year,
                      "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"},
        ))
    except Exception:
        continue

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Parsed {len(documents)} documents → {len(chunks)} chunks\n")

# ── Shared prompt ─────────────────────────────────────────────────────────────

prompt = PromptTemplate.from_template(
    "You are a biomedical research assistant. Use ONLY the provided context to answer the question.\n"
    "If the context does not contain enough information, say 'I don't have enough information in the retrieved abstracts to answer this question.'\n"
    "Do not use prior knowledge outside the context.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

# ── Run benchmark ─────────────────────────────────────────────────────────────

results = []

for emb_model in EMBEDDING_MODELS:
    print(f"\n=== Embedding: {emb_model} ===")
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', emb_model)
    persist_dir = f"./chroma_db/benchmark_{safe_name}"

    print("  Building vector DB...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=emb_model)
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    except Exception as e:
        print(f"  Failed to build DB: {e}")
        continue

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Build chain with RunnableParallel — same pattern as ask.py
    retrieval_step = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough()
    )
    answer_chain = (
        RunnableLambda(lambda x: {
            "context": "\n\n".join(doc.page_content for doc in x["docs"]),
            "question": x["question"],
        })
        | prompt
        | StrOutputParser()
    )

    for llm_model in LLM_MODELS:
        print(f"\n  LLM: {llm_model}")
        try:
            llm = OllamaLLM(model=llm_model)
        except Exception as e:
            print(f"  Failed to load LLM: {e}")
            continue

        chain = retrieval_step | RunnableParallel(
            answer=answer_chain | llm | StrOutputParser(),
            sources=RunnableLambda(lambda x: x["docs"]),
        )

        for question in QUESTIONS:
            print(f"    Q: {question[:60]}...")
            start = time.time()
            try:
                result   = chain.invoke(question)
                response = result["answer"]
                elapsed  = time.time() - start
                cited_pmids = list({
                    doc.metadata.get("pmid", "") for doc in result["sources"]
                    if doc.metadata.get("pmid")
                })
                print(f"       ({elapsed:.1f}s) sources: {cited_pmids}")
            except Exception as e:
                response    = f"ERROR: {e}"
                elapsed     = -1
                cited_pmids = []
                print(f"       Error: {e}")

            results.append({
                "embedding_model":  emb_model,
                "llm_model":        llm_model,
                "question":         question,
                "response":         response[:300],
                "response_time_s":  round(elapsed, 2),
                "retrieved_pmids":  "|".join(cited_pmids),
            })

# ── Save results ──────────────────────────────────────────────────────────────

output_file = "benchmark_results.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["embedding_model", "llm_model", "question",
                    "response", "response_time_s", "retrieved_pmids"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmark complete. Results saved to {output_file}")
print(f"Total runs: {len(EMBEDDING_MODELS)} embeddings × {len(LLM_MODELS)} LLMs × {len(QUESTIONS)} questions = {len(results)}")
