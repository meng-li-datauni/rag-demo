"""RAG pipeline: load PDFs, chunk, embed locally, retrieve, answer with Claude."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

DOCS_DIR = Path(__file__).parent / "docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
TOP_K = 3

NO_ANSWER = "I don't know based on the provided documents."

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context excerpts below, which come from internal "
     "company documents. Do not use outside knowledge. If the context does not contain the "
     "answer, reply exactly: \"" + NO_ANSWER + "\" "
     "Keep answers short and end with the source file name(s) in square brackets.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def load_pdfs(docs_dir: Path = DOCS_DIR) -> list[Document]:
    docs = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        for page_num, page in enumerate(PdfReader(pdf_path).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text,
                                     metadata={"source": pdf_path.name, "page": page_num}))
    if not docs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir} - run generate_docs.py first")
    return docs


class RagPipeline:
    def __init__(self, docs_dir: Path = DOCS_DIR, top_k: int = TOP_K):
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(load_pdfs(docs_dir))
        self.vector_store = InMemoryVectorStore.from_documents(
            chunks, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        self.llm = ChatAnthropic(model=LLM_MODEL, max_tokens=4096)
        self.num_chunks = len(chunks)

    def ask(self, question: str, config: RunnableConfig | None = None
            ) -> tuple[str, list[Document]]:
        # Wrapped in a runnable so retrieval + LLM call show up as one trace in LangSmith
        # and callbacks (e.g. usage tracking) reach the LLM call.
        return RunnableLambda(self._ask, name="rag").invoke(question, config)

    def _ask(self, question: str) -> tuple[str, list[Document]]:
        retrieved = self.retriever.invoke(question)
        context = "\n\n".join(
            f"[{d.metadata['source']}]\n{d.page_content}" for d in retrieved
        )
        response = self.llm.invoke(PROMPT.format_messages(context=context, question=question))
        return response.text, retrieved
