import os
import re
import json
from pathlib import Path

# Dictionary mapping CUDA API/types to HIP equivalents
CUDA_TO_HIP_MAPPING = {
    # Memory Management
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyAsync": "hipMemcpyAsync",
    "cudaMemcpyToSymbol": "hipMemcpyToSymbol",
    "cudaMemcpyFromSymbol": "hipMemcpyFromSymbol",
    
    # Device Management & Synchronization
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    
    # Function Qualifiers (Often identical in HIP, but still needed for migration awareness)
    "__global__": "__global__",
    "__device__": "__device__",
    "__host__": "__host__",
    
    # Headers
    "cuda_runtime.h": "hip/hip_runtime.h",
    
    # Namespaces
    "cooperative_groups::": "cooperative_groups::", # ROCm uses identical namespace
    "thrust::": "hipthrust::", # hipthrust is the HIP port of thrust
    
    # Atomics
    "atomicAdd": "atomicAdd",
    "atomicCAS": "atomicCAS",
}

def parse_cuda_file(filepath: str) -> dict:
    """
    Parses a CUDA file using regex to identify CUDA-specific patterns and 
    suggests HIP equivalents.
    
    Returns a dictionary structured as:
    {
      "filename": "...",
      "total_lines": ...,
      "cuda_calls_found": [...],
      "total_conversions_needed": ...
    }
    """
    filepath_obj = Path(filepath)
    filename = filepath_obj.name
    
    if not filepath_obj.exists():
        return {"error": f"File {filepath} not found."}
        
    cuda_calls_found = []
    total_lines = 0
    
    # Compile regex patterns for efficiency
    
    # 1. Simple word/string matching for things in our dictionary
    # Sort keys by length descending to match longest first (e.g., cudaMemcpyAsync before cudaMemcpy)
    sorted_keys = sorted(CUDA_TO_HIP_MAPPING.keys(), key=len, reverse=True)
    
    pattern_parts = []
    for k in sorted_keys:
        # For namespaces and headers, we don't necessarily want strict word boundaries on both sides
        if k.endswith("::") or k.endswith(".h"):
            pattern_parts.append(re.escape(k))
        else:
            # \b matches word boundaries, making sure we don't match 'my_cudaMalloc' partially
            pattern_parts.append(r'\b' + re.escape(k) + r'\b')
            
    dict_pattern = re.compile(r'(' + '|'.join(pattern_parts) + r')')
    
    # 2. Kernel launch syntax <<< ... >>>
    launch_pattern = re.compile(r'<<<(.*?)>>>')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line_content in enumerate(f, start=1):
            total_lines += 1
            
            # Find matches for dictionary items
            for match in dict_pattern.finditer(line_content):
                call = match.group(0)
                cuda_calls_found.append({
                    "call": call,
                    "line": line_num,
                    "hip_equivalent": CUDA_TO_HIP_MAPPING.get(call, call)
                })
                
            # Find matches for kernel launch syntax <<<...>>>
            if launch_pattern.search(line_content):
                cuda_calls_found.append({
                    "call": "<<<...>>>",
                    "line": line_num,
                    "hip_equivalent": "hipLaunchKernelGGL"
                })

    report = {
        "filename": filename,
        "total_lines": total_lines,
        "cuda_calls_found": cuda_calls_found,
        "total_conversions_needed": len(cuda_calls_found)
    }
    
    return report
