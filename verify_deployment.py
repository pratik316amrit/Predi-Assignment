import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the directory
load_dotenv()

try:
    print("1. Testing Imports...")
    from rag_pipeline_API import VehicleSpecRAG
    import llm_engine
    print("   Imports successful.")
except ImportError as e:
    print(f"   Import failed: {e}")
    sys.exit(1)

def test_logic():
    print("\n2. Initializing RAG (API Mode)...")
    pdf_path = os.path.join("PDFs", "Assignment-specs-extraction 1.pdf")
    hf_key = os.getenv("HF_API_KEY")
    
    if not hf_key:
        print("   HF_API_KEY missing.")
        return

    try:
        rag = VehicleSpecRAG(pdf_path, hf_key)
        
        # Check index (mocking the app logic)
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name=rag.embedding_model_name)
        rag.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=rag.persist_directory,
        )
        rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 2})
        print("   RAG initialized.")
        
        print("\n3. Testing Retrieval & Extraction...")
        query = "whats the Torque for brake caliper anchor plate bolts"
        
        docs = rag.retrieve_with_references(query)
        if not docs:
            print("   Retrieval returned no docs.")
            return

        print(f"   Retrieved {len(docs)} docs.")
        context = "\n---\n".join([d.page_content for d in docs])
        
        print("   calling LLM (API)...")
        results = llm_engine.extract_with_llama_api(context, query)
        
        print(f"   Results: {results}")

    except Exception as e:
        print(f"   Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logic()
