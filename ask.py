from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import logging
import time

logging.basicConfig(
    filename="ask.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

persist_dir = input("Enter DB path to use (default: ./chroma_db): ").strip()
if not persist_dir:
    persist_dir = "./chroma_db"

if not os.path.exists(persist_dir):
    print(f"Path not found: {persist_dir}")
    logging.error(f"DB path not found: {persist_dir}")
    exit(1)

print(f"Loading DB from: {persist_dir}")
logging.info(f"Loading DB from: {persist_dir}")

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
except Exception as e:
    print(f"Error loading DB: {e}")
    logging.error(f"Failed to load DB: {e}")
    exit(1)

llm = OllamaLLM(model="llama3.2")
retriever = vectorstore.as_retriever()

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

while True:
    query = input("\nEnter your question (quit: q): ").strip()
    if query.lower() == "q":
        break
    if not query:
        continue
    logging.info(f"Question: {query}")
    start = time.time()
    try:
        response = chain.invoke(query)
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Chain invoke failed: {e}")
        continue
    elapsed = time.time() - start
    print(f"\nAnswer: {response}")
    print(f"[{elapsed:.1f}s]")
    logging.info(f"Response time: {elapsed:.1f}s")
