import os
import json
import requests
import re
from pathlib import Path
from dotenv import load_dotenv

def migrate_kernel(filepath: str, manifest_filepath: str, mode: str = "code_only"):
    load_dotenv()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment variables.")
        return None
        
    filepath_obj = Path(filepath)
    manifest_obj = Path(manifest_filepath)
    
    if not filepath_obj.exists() or not manifest_obj.exists():
        print("Error: CUDA file or manifest file not found.")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        cuda_code = f.read()
        
    with open(manifest_filepath, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
        
    # Gemma model used for AMD Hackathon ACT II Gemma Prize eligibility
    model_name = "accounts/fireworks/models/gemma-3-27b-it"
    
    if mode == "code_only":
        prompt_instruction = "Output ONLY the raw HIP code in a single code block. Do NOT provide any reasoning, explanation, or commentary before or after the code block."
        max_tokens = 8192
    else:
        prompt_instruction = "First, briefly explain your key conversion decisions (under 150 words). Then, output the full HIP code in a separate clearly marked code block wrapped in ```cpp ... ``` tags."
        max_tokens = 8192

    prompt = (
        "You are an expert GPU programming engineer migrating CUDA code to HIP.\n"
        "Please convert the following CUDA C++ code to HIP.\n"
        "Here are the specific CUDA calls identified by our AST parser that need conversion:\n"
        f"{json.dumps(manifest_data.get('cuda_calls_found', []), indent=2)}\n\n"
        "Requirements:\n"
        "1. Output valid, compilable HIP code.\n"
        "2. Replace <cuda_runtime.h> with <hip/hip_runtime.h> if it exists.\n"
        "3. Ensure all hip* function replacements are correct.\n"
        f"4. {prompt_instruction}\n\n"
        "Original CUDA Code:\n"
        "```cpp\n"
        f"{cuda_code}\n"
        "```\n"
    )
    
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    estimated_input_tokens = len(prompt) // 4
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        
        if response.status_code == 200:
            full_message = ""
            prompt_tokens = estimated_input_tokens
            completion_tokens = 0
            total_tokens = 0
            
            # Read streaming response chunks to prevent connection timeouts
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line_str[6:])
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_message += content
                            
                            # Fireworks sends usage stats in chunks sometimes
                            if "usage" in chunk_data and chunk_data["usage"]:
                                usage = chunk_data["usage"]
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                                total_tokens = usage.get("total_tokens", total_tokens)
                        except json.JSONDecodeError:
                            pass
                            
            if total_tokens == 0:
                completion_tokens = len(full_message) // 4
                total_tokens = prompt_tokens + completion_tokens
                
            message = full_message
            
            code_match = re.search(r'```(?:cpp|c\+\+)?\n(.*?)```', message, re.DOTALL)
            if code_match:
                hip_code = code_match.group(1).strip()
                reasoning = message.replace(code_match.group(0), "").strip()
            else:
                # If model ignored instruction and just dumped code without markdown or reason
                hip_code = message.strip()
                if hip_code.startswith("```"):
                    hip_code = hip_code.lstrip("```cpp\n").lstrip("```c++\n").lstrip("```\n").rstrip("```")
                reasoning = ""
                
            converted_dir = filepath_obj.parent.parent / 'converted'
            filename_base = filepath_obj.name
            
            hip_filepath = converted_dir / f"{filename_base}.hip"
            reasoning_filepath = converted_dir / f"{filename_base}_reasoning.txt"
            
            with open(hip_filepath, 'w', encoding='utf-8') as f:
                f.write(hip_code)
                
            if reasoning and mode != "code_only":
                with open(reasoning_filepath, 'w', encoding='utf-8') as f:
                    f.write(reasoning)
                
            return {
                "hip_file": str(hip_filepath),
                "reasoning_file": str(reasoning_filepath) if (reasoning and mode != "code_only") else None,
                "hip_code": hip_code,
                "reasoning": reasoning,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Exception during API call: {e}")
        return None
