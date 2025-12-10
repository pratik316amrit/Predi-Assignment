import os
import re
import json
import torch
import warnings
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants and Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_API_KEY not found in environment.")

# API Based Extraction
def extract_with_llama_api(context: str, query: str) -> list:
    """
    Extracts vehicle specifications using the Hugging Face Inference API.
    """
    client = InferenceClient(api_key=HF_TOKEN)
    
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
            model=MODEL_ID, 
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
        return _parse_json_response(generated_text)

    except Exception as e:
        print(f"    API Request Error: {e}")
        return []

# Local Based Extraction
_local_model = None
_local_tokenizer = None

def _load_local_model():
    """Lazily loads the local model and tokenizer."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        try:
            print("Loading local model (this may take a while)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device}")

            _local_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
            _local_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
            if device == "cpu":
                _local_model.to(device)
                
            print("   Model loaded successfully.")
        except Exception as e:
            print(f"   Error loading local model: {e}")
            raise e

def extract_with_llama_local(context: str, query: str) -> list:
    """
    Extracts vehicle specifications using the locally loaded Llama model.
    """
    _load_local_model()
    
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

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    try:
        inputs = _local_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_local_model.device)

        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )

        generated_text = _local_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        return _parse_json_response(generated_text)

    except Exception as e:
        print(f"   Generation Error: {e}")
        return []

# Utility: JSON Parsing
def _parse_json_response(text: str) -> list:
    """Parses JSON list from the LLM output."""
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    
    # Try finding a list
    json_match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding a single object
    json_match_obj = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
    if json_match_obj:
        try:
            return [json.loads(json_match_obj.group(0))]
        except json.JSONDecodeError:
            pass
            
    return []
