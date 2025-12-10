"""
FINAL VERIFICATION: Llama-3.1 Extraction with Raw Vector Retrieval
"""

import os
import re
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
from rag_pipeline import VehicleSpecRAG

# 1. Setup Llama 3.1 Model
print("=" * 80)
print("INITIALIZING Llama-3.1-8B-Instruct...")
print("=" * 80)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
hf_token = os.getenv("HF_API_KEY")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model.to(device)

    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


# 2. Extraction Logic

def extract_with_llama(context: str, query: str) -> list:
    """
    Feeds the raw retrieved text to Llama 3.1 for extraction.
    """

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

    user_content = f"""CONTEXT:
    {context}

    QUERY: {query}"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # Cleanup
        cleaned_text = generated_text.replace("```json", "").replace("```", "").strip()

        # Parse JSON list
        json_match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        # Or single JSON object
        json_match_obj = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if json_match_obj:
            return [json.loads(json_match_obj.group(0))]

        return []

    except Exception as e:
        print(f"   Generation Error: {e}")
        return []


# 3. Main Test Loop

def test_full_pipeline():
    print("=" * 80)
    print("FULL PIPELINE TEST (Raw Vector + Llama 3.1)")
    print("=" * 80)

    pdf_path = os.path.join("PDFs", "Assignment-specs-extraction 1.pdf")
    hf_api_key = os.getenv("HF_API_KEY")

    # Initialize RAG
    rag = VehicleSpecRAG(pdf_path, hf_api_key)

    # 🔁 Cache-aware index logic
    if rag.check_index_exists():
        print("Using existing Chroma index (no re-chunking).")
        embeddings = HuggingFaceEmbeddings(model_name=rag.embedding_model_name)
        rag.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=rag.persist_directory,
        )
        rag.retriever = rag.vectorstore.as_retriever(search_kwargs={"k": 15})
    else:
        print("Index missing or outdated: rebuilding (convert -> chunk -> vectorize)...")
        rag.convert_pdf_to_markdown()
        rag.create_chunks()
        rag.initialize_retriever(k=15)

    # Test Cases
    queries = [
        "whats the Torque for brake caliper anchor plate bolts",
        "Whats the Brake disc minimum thickness?",
        "Brake pad minimum thickness",
        "brake disc shield bolts torque",
        "whats the specs for Additive Friction modifier XL-3",
        "whats the part number for Differential side gear?",
        "whats the specs for Motorcraft® SAE 75W-140 "
    ]

    final_results = []

    for query in queries:
        print(f"\nQUERY: '{query}'")

        # 1. RETRIEVE
        docs = rag.retrieve_with_references(query)

        if not docs:
            print("   No documents found.")
            continue

        top_docs = docs[:2]

        # Collect raw page numbers from metadata
        raw_pages = {
            d.metadata.get("page")
            for d in top_docs
            if d.metadata.get("page") is not None
        }

        # Hard-coded adjustment:
        # - If page <= 1 or None, treat as 1
        # - Else, subtract 1 (viewer shows one less than metadata)
        pages = sorted(
            {1 if (p is None or p <= 1) else p - 1 for p in raw_pages}
        )

        print(f"   Pages used for this query (adjusted): {pages}")

        # Combine top docs for context
        context = "\n---\n".join([d.page_content for d in top_docs])

        print(f"   Feeding {len(top_docs)} chunks to Llama...")

        # 2. EXTRACT
        extracted_data = extract_with_llama(context, query)

        if extracted_data:
            for item in extracted_data:
                item["pages"] = pages

            print("   Extraction Successful!")
            print(f"      Result: {json.dumps(extracted_data, indent=2)}")
            final_results.extend(extracted_data)
        else:
            print("   Llama returned empty list or invalid JSON.")

    # 4. Final Output
    print("\n" + "=" * 80)
    print("FINAL COLLECTED DATA")
    print("=" * 80)
    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    test_full_pipeline()
