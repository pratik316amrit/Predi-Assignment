import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional

import pymupdf4llm
from huggingface_hub import InferenceClient

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Bump this whenever you change chunking / vectorization logic
INDEX_VERSION = "api_v3_pageaware_chunking"


# Simple HF Embeddings API wrapper (no local download)

class HFAPIEmbeddings:
    """
    Minimal embedding class compatible with Chroma+LangChain,
    but uses Hugging Face Inference via huggingface_hub.InferenceClient.

    IMPORTANT:
    - We DO NOT set provider="hf-inference" here, because that requires
      Inference Providers permissions (paid).
    - Instead, we bind the model directly and rely on serverless inference.
    """

    def __init__(self, api_key: str, model_name: str):
        self.api_key = "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"
        self.model_name = model_name
        # Bind this client to the embedding model directly
        self.client = InferenceClient(
            model=self.model_name,
            api_key=self.api_key,  # or token=self.api_key depending on hf-hub version
        )

    @staticmethod
    def _ensure_1d(vec):
        """
        HF feature_extraction may return either:
        - [dim]                         (already a sentence embedding)
        - [[dim], [dim], ...]          (per-token embeddings)
        We convert everything to a single 1D vector by mean-pooling if needed.
        """
        # 1. Handle None
        if vec is None:
            return []
            
        # 2. Convert NumPy array to list (fixes the ValueError)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # 3. Now safe to check if empty
        if not vec:
            return []

        # 4. Process as before
        if isinstance(vec[0], (float, int)):
            # Already 1D
            return [float(x) for x in vec]

        # Assume 2D: list of [dim]
        length = len(vec)
        dim = len(vec[0])
        out = [0.0] * dim
        for token_vec in vec:
            for j, val in enumerate(token_vec):
                out[j] += float(val)
        return [v / length for v in out]

    def _embed_single(self, text: str) -> List[float]:
        # Use the HF InferenceClient feature_extraction API bound to this model
        result = self.client.feature_extraction(text)
        return self._ensure_1d(result)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

# Main RAG class

class VehicleSpecRAG:
    """
    RAG pipeline for extracting vehicle specifications from an automotive
    service manual PDF using ONLY API-based models.

    - PDF → markdown page-wise (local)
    - Page-aware, table-aware chunking (local)
    - Embeddings: HF Inference API (BAAI/bge-m3 or similar)
    - Vector store: Chroma (local DB)
    """

    def __init__(self, pdf_path: str, hf_api_key: Optional[str] = None):
        self.pdf_path = pdf_path
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
        if not self.hf_api_key:
            raise ValueError("HF_API_KEY not found in environment and not provided explicitly.")

        # Page-wise markdown: [{"text": "...", "metadata": {"page": ...}}, ...]
        self.markdown_pages: List[Dict[str, Any]] = []

        # Chroma persistence
        self.persist_directory = "./chroma_db_api"
        self.metadata_path = os.path.join(self.persist_directory, "metadata_api.json")

        # Embedding model name (HF Inference)
        self.embedding_model_name = "BAAI/bge-m3"

        # Chunks & vectorstore
        self.chunks: List[Document] = []
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None

        # Versioning
        self.index_version = INDEX_VERSION

    # File hash and index metadata (for caching)

    def _get_file_hash(self) -> str:
        """Compute an MD5 hash of the PDF file content."""
        hasher = hashlib.md5()
        with open(self.pdf_path, "rb") as f:
            buf = f.read(65536)
            while buf:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def check_index_exists(self) -> bool:
        """
        Check whether a Chroma index exists and matches:
        - current PDF file hash
        - current INDEX_VERSION
        - same filename
        """
        if not os.path.exists(self.persist_directory) or not os.path.exists(self.metadata_path):
            return False

        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            current_hash = self._get_file_hash()
            filename_ok = metadata.get("filename") == os.path.basename(self.pdf_path)
            hash_ok = metadata.get("hash") == current_hash
            version_ok = metadata.get("index_version") == self.index_version

            if filename_ok and hash_ok and version_ok:
                print("Existing API-based index is up-to-date (file + version match).")
                return True

            print("API index exists but is outdated or for a different file/version.")
            print(f"   filename_ok={filename_ok}, hash_ok={hash_ok}, version_ok={version_ok}")
            return False

        except Exception as e:
            print(f"Error while checking API index metadata: {e}")
            return False

    def save_index_metadata(self):
        """Save hash + filename + index_version for future index reuse."""
        os.makedirs(self.persist_directory, exist_ok=True)
        meta = {
            "filename": os.path.basename(self.pdf_path),
            "hash": self._get_file_hash(),
            "index_version": self.index_version,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    # PDF to Markdown (page-wise)

    def convert_pdf_to_markdown(self):
        """
        Convert the PDF to markdown using pymupdf4llm, with page_chunks=True.
        """
        print("Converting PDF to markdown (page-wise)...")
        self.markdown_pages = pymupdf4llm.to_markdown(self.pdf_path, page_chunks=True)
        print(f"   Extracted {len(self.markdown_pages)} pages.")

    # Page-aware, table-aware chunking

    def create_chunks(self):
        """
        Create page-aware, table-aware chunks:
        - Never mixes content across pages
        - Groups table lines (with '|') into table blocks
        - Splits text blocks into overlapping chunks
        """
        if not self.markdown_pages:
            raise ValueError("Markdown pages are empty. Run convert_pdf_to_markdown() first.")

        print("Creating page-aware chunks (API version)...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,    # roughly ~400–700 tokens
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        all_chunks: List[Document] = []

        def split_page_into_blocks(page_text: str):
            """
            Split a page into ('table'|'text', block_text) blocks.
            Consecutive lines containing '|' 2+ times are treated as tables.
            """
            lines = page_text.splitlines()
            blocks = []
            current_lines = []
            current_type = None  # 'table' or 'text'

            def is_table_line(ln: str) -> bool:
                return ln.count("|") >= 2

            for ln in lines:
                ln_type = "table" if is_table_line(ln) else "text"

                if current_type is None:
                    current_type = ln_type
                    current_lines = [ln]
                elif ln_type == current_type:
                    current_lines.append(ln)
                else:
                    blocks.append((current_type, "\n".join(current_lines).strip()))
                    current_type = ln_type
                    current_lines = [ln]

            if current_lines:
                blocks.append((current_type, "\n".join(current_lines).strip()))

            return [(t, b) for t, b in blocks if b.strip()]

        section_pattern = r"\b(\d{3}-\d{2})\b"  # e.g. "206-03"

        num_pages = len(self.markdown_pages)
        for idx, page_info in enumerate(self.markdown_pages):
            meta = page_info.get("metadata", {}) or {}

            # Use PDF's own page number if provided, else idx+1
            if "page" in meta:
                page_num = int(meta["page"])
            else:
                page_num = idx + 1

            page_text = page_info.get("text", "") or ""

            print(f"   Processing page {page_num}/{num_pages}...")

            blocks = split_page_into_blocks(page_text)

            for block_type, block_text in blocks:
                if not block_text.strip():
                    continue

                section_match = re.search(section_pattern, block_text)
                section_number = section_match.group(1) if section_match else None

                base_metadata: Dict[str, Any] = {
                    "page": page_num,
                    "type": block_type,
                }
                if section_number:
                    base_metadata["section_number"] = section_number

                if block_type == "table":
                    doc = Document(
                        page_content=block_text,
                        metadata={**base_metadata, "contains_table": True},
                    )
                    all_chunks.append(doc)
                else:
                    split_texts = text_splitter.split_text(block_text)
                    for txt in split_texts:
                        doc = Document(
                            page_content=txt,
                            metadata=base_metadata.copy(),
                        )
                        all_chunks.append(doc)

        self.chunks = all_chunks
        print(f"   Created {len(self.chunks)} chunks total (API).")
        print("      - with section numbers:",
              sum(1 for c in self.chunks if c.metadata.get("section_number")))
        print("      - with tables:",
              sum(1 for c in self.chunks if c.metadata.get("contains_table")))

    # Vector store and retriever (Chroma + HF API embeddings)

    def initialize_retriever(self, k: int = 15):
        """
        Build embeddings using HF API (BAAI/bge-m3) and create a Chroma vector store.
        """
        if not self.chunks:
            raise ValueError("Chunks not created. Run create_chunks() first.")

        print("Initializing embeddings (HF API) and Chroma vector store...")
        embeddings = HFAPIEmbeddings(
            api_key=self.hf_api_key,
            model_name=self.embedding_model_name,
        )

        print("   Preparing metadata for Chroma...")
        for chunk in self.chunks:
            if "page" in chunk.metadata:
                page_int = int(chunk.metadata["page"])
                chunk.metadata["page"] = page_int
                chunk.metadata["page_str"] = str(page_int)

        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
        )

        print("   Creating dense retriever (Chroma + HF API)...")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.save_index_metadata()
        print("   API-based retriever initialized.")

    # Retrieval helper for test_API.py

    def retrieve_with_references(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k documents (chunks) for a given query using the retriever.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Run initialize_retriever() first.")

        try:
            if getattr(self.retriever, "search_kwargs", None) is not None:
                self.retriever.search_kwargs["k"] = k
        except Exception:
            pass

        docs = self.retriever.invoke(query)
        if isinstance(docs, Document):
            docs = [docs]

        return docs
