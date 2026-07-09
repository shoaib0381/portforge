import os
import json
from pathlib import Path
from ast_parser import parse_cuda_file

def main():
    # Define absolute paths based on this file's location
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    raw_dir = project_root / 'kernels' / 'raw'
    converted_dir = project_root / 'kernels' / 'converted'
    
    # Ensure converted directory exists
    converted_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .cu files from the raw directory
    cu_files = list(raw_dir.glob('*.cu'))
    
    if not cu_files:
        print(f"No .cu files found in {raw_dir}")
        return
        
    print(f"Found {len(cu_files)} CUDA files. Starting parser...\n")
    
    # Process each CUDA file
    for cu_file in cu_files:
        report = parse_cuda_file(str(cu_file))
        
        if "error" in report:
            print(f"Error processing {cu_file.name}: {report['error']}")
            continue
            
        # Save JSON report manifest
        output_filename = f"{cu_file.name}_manifest.json"
        output_path = converted_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        # Print summary output for the file
        filename = report['filename']
        conversions = report['total_conversions_needed']
        print(f"- {filename}: {conversions} conversions needed")
        
    print("\nParsing complete. Manifests saved to /kernels/converted/")

if __name__ == "__main__":
    main()
