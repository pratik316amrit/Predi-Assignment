import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import RAG pipelines and LLM engine (reusing existing logic)
from rag_pipeline import VehicleSpecRAG as LocalRAG
from rag_pipeline_API import VehicleSpecRAG as ApiRAG
import llm_engine

load_dotenv()

app = FastAPI(title="SpecExtractor API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global RAG Instances (Lazy loaded)
rag_systems = {
    "API": None,
    "Local": None
}

class QueryRequest(BaseModel):
    query: str
    mode: str = "API"

def get_or_create_rag(mode: str):
    """Retrieves or initializes the RAG system for the specified mode."""
    if rag_systems[mode]:
        return rag_systems[mode]
    
    pdf_path = os.path.join("PDFs", "Assignment-specs-extraction 1.pdf")
    hf_api_key = os.getenv("HF_API_KEY")
    
    if not hf_api_key:
        raise HTTPException(status_code=500, detail="HF_API_KEY missing in environment.")
    
    print(f"Initializing {mode} RAG System...")
    
    if mode == "API":
        rag = ApiRAG(pdf_path, hf_api_key)
    elif mode == "Local":
        rag = LocalRAG(pdf_path, hf_api_key)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    # Check/Build Index
    if rag.check_index_exists():
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        embeddings = HuggingFaceEmbeddings(model_name=rag.embedding_model_name)
        rag.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=rag.persist_directory,
        )
        rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 15})
    else:
        print("Building Index...")
        rag.convert_pdf_to_markdown()
        rag.create_chunks()
        rag.initialize_retriever(k=15)
        
    rag_systems[mode] = rag
    return rag

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/api/query")
async def process_query(request: QueryRequest):
    try:
        rag = get_or_create_rag(request.mode)
        
        # Retrieve
        docs = rag.retrieve_with_references(request.query)
        
        if not docs:
            return JSONResponse(content={"message": "No relevant documents found.", "data": []})
        
        top_docs = docs[:2]
        context = "\n---\n".join([d.page_content for d in top_docs])
        pages = sorted({d.metadata.get("page", 0) for d in top_docs})
        
        # Extract
        if request.mode == "API":
            data = llm_engine.extract_with_llama_api(context, request.query)
        else:
            data = llm_engine.extract_with_llama_local(context, request.query)
            
        return JSONResponse(content={
            "message": "Success", 
            "data": data,
            "pages": pages,
            "context_preview": context[:200] + "..."
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)
