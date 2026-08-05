"""
Vectorstore retriever setup.

The original prototype referenced a bare `retriever` object without ever
defining it. This module builds one: it chunks a small set of source
documents, embeds them with Anthropic-compatible embeddings via
HuggingFace, and stores them in a local Chroma index.

Swap `SOURCE_URLS` / `load_documents()` for your own ingestion logic
(PDFs, a database, an existing index, etc.) as needed.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Example seed sources — replace with your own corpus.
SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]


def load_documents(urls=None):
    """Load and split documents into retriever-sized chunks."""
    urls = urls or SOURCE_URLS
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    return splitter.split_documents(docs_list)


def build_retriever(persist_directory: str = ".chroma_db", k: int = 4):
    """Build (or load) a Chroma vectorstore and return it as a retriever."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    splits = load_documents()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="crag-chroma",
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# Built once at import time and reused by nodes.py, matching the
# single-shared-instance pattern used for `llm` in config.py.
retriever = build_retriever()
