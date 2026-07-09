import argparse
from pathlib import Path
from migration_agent import migrate_kernel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="matrixMul.cu")
    parser.add_argument("--mode", type=str, default="code_only")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_file = project_root / 'kernels' / 'raw' / args.file
    manifest_file = project_root / 'kernels' / 'converted' / f"{args.file}_manifest.json"
    
    print(f"Testing migration on: {raw_file.name} (Mode: {args.mode})")
    
    result = migrate_kernel(str(raw_file), str(manifest_file), mode=args.mode)
    
    if result:
        print("\n--- MIGRATION SUCCESSFUL ---")
        print(f"HIP File Saved To: {result['hip_file']}")
        if result['reasoning_file']:
            print(f"Reasoning File Saved To: {result['reasoning_file']}")
        
        # Calculate cost. Fireworks open source models (<=16B) typically cost $0.20 per 1M tokens.
        cost_per_1m = 0.20
        total_tokens = result['total_tokens']
        cost = (total_tokens / 1_000_000) * cost_per_1m
        
        print(f"\n--- TOKEN USAGE & COST ---")
        print(f"Prompt Tokens:     {result['prompt_tokens']}")
        print(f"Completion Tokens: {result['completion_tokens']}")
        print(f"Total Tokens:      {total_tokens}")
        print(f"Estimated Cost:    ${cost:.6f} USD")
        
    else:
        print("\n--- MIGRATION FAILED ---")

if __name__ == "__main__":
    main()
