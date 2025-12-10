import os
import re
import json
import warnings
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
from rag_pipeline_API import VehicleSpecRAG

print("=" * 80)
print("INITIALIZING Llama-3.1-8B-Instruct...")
print("=" * 80)

hf_token = "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"
if not hf_token:
    raise ValueError("HF_API_KEY not found in environment.")

client = InferenceClient(api_key=hf_token)

print("HF InferenceClient ready.")

def extract_with_llama_api(context: str, query: str) -> list:
    system_content = """You are an expert automotive data extraction assistant.
    Analyze the provided text context and extract the exact specification requested by the user.

    RULES:
    1. Output ONLY a valid JSON list.
    2. Do not output any conversation, markdown backticks, or explanations.
    3. If the answer is not in the text, return an empty list [].
    4. Pay attention to tables. Align headers with values carefully.

    JSON FORMAT:
    [
      {
        "component": "Exact component name from text",
        "spec_type": "Torque" or "Dimension" or "Capacity" or "Fluid Capacity" or "Part number",
        "value": "Numeric value (e.g., '25') or if the Part number doesnt say anything logically numerical, then put None or if the specification are string them put that specification string value only if that string is a combination of characters and numbers like T6675-HYUJ",
        "unit": "Unit (e.g., 'Nm', 'mm', 'L') or null/None if the context does not specify a meaningful unit (especially for part numbers)"
      }
    ]"""


    user_content = f"CONTEXT:\n{context}\n\nQUERY: {query}"

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct", 
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            stream=False
        )

        generated_text = response.choices[0].message.content
        cleaned_text = generated_text.replace("```json", "").replace("```", "").strip()
        
        json_match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        json_match_obj = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if json_match_obj:
            return [json.loads(json_match_obj.group(0))]
        
        return []

    except Exception as e:
        print(f"    API Request Error: {e}")
        return []

def test_full_pipeline_api():
    print("=" * 80)
    print("FULL PIPELINE TEST (API-only RAG + Llama 3.1)")
    print("=" * 80)

    pdf_path = os.path.join("PDFs", "Assignment-specs-extraction 1.pdf")
    hf_api_key = os.getenv("HF_API_KEY")

    rag = VehicleSpecRAG(pdf_path, hf_api_key)

    if rag.check_index_exists():
        print("Using existing API-based Chroma index.")
        embeddings = HuggingFaceEmbeddings(model_name=rag.embedding_model_name)
        rag.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=rag.persist_directory,
        )
        rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 15})
    else:
        print("Building index...")
        rag.convert_pdf_to_markdown()
        rag.create_chunks()
        rag.initialize_retriever(k=15)

    queries = [
        "whats the Torque for brake caliper anchor plate bolts",
        "Whats the Brake disc minimum thickness?",
        "Brake pad minimum thickness",
        "brake disc shield bolts torque",
        "whats the specs for Additive Friction modifier XL-3",
        "whats the part number for Differential side gear?"
    ]

    for query in queries:
        print(f"\nQUERY: '{query}'")
        docs = rag.retrieve_with_references(query)
        if not docs:
            print("    No documents found.")
            continue
            
        top_docs = docs[:2]
        
        raw_pages = {
            d.metadata.get("page")
            for d in top_docs
            if d.metadata.get("page") is not None
        }

        pages = sorted(
            {1 if (p is None or p <= 1) else p - 1 for p in raw_pages}
        )

        context = "\n---\n".join([d.page_content for d in top_docs])
        print(f"    Feeding {len(top_docs)} chunks to Llama...")
        
        data = extract_with_llama_api(context, query)
        
        if data:
            for item in data:
                item["pages"] = pages
            print(f"    Result: {json.dumps(data, indent=2)}")
        else:
             print("    Llama returned empty list.")

if __name__ == "__main__":
    test_full_pipeline_api()