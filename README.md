<img width="1907" height="857" alt="image" src="https://github.com/user-attachments/assets/5e32dc14-2a87-43f4-a2bb-03de5de58bd5" />
<img width="1893" height="863" alt="image" src="https://github.com/user-attachments/assets/6afb6b98-4017-4ae8-a0c8-50ddec558107" />
<img width="1895" height="862" alt="image" src="https://github.com/user-attachments/assets/300d4fff-4f8c-481f-a3d4-c369deee90d5" />

<img width="1898" height="858" alt="image" src="https://github.com/user-attachments/assets/bdcaec94-110d-47e7-867e-34d24b76fd63" />
<img width="1896" height="718" alt="image" src="https://github.com/user-attachments/assets/e6eb455e-4eb5-4cfb-9cae-63887c9e197c" />


# PortForge

**AI-powered CUDA-to-ROCm/HIP migration agent — automated, validated, and benchmarked on AMD MI300X.**

## What is PortForge?

PortForge is an agentic pipeline that takes existing NVIDIA CUDA code and automatically migrates it to AMD's ROCm/HIP platform. Unlike mechanical translation tools (like AMD's own `hipify-clang`), PortForge uses an AI reasoning model to understand code semantics, generate valid HIP equivalents, compile them on real AMD hardware, fix errors automatically, and benchmark the results — all without manual developer intervention.

Built for the **AMD Developer Hackathon: ACT II** (Unicorn Track).

## The Problem

NVIDIA's CUDA ecosystem has a 15+ year head start and millions of developers fluent in it. AMD's ROCm platform is hardware-competitive, but developers evaluating AMD hardware hit a wall: migrating existing CUDA codebases is slow, manual, and error-prone. Existing tools like `hipify-clang` do basic syntax translation but don't compile, validate, or benchmark — they leave the hard part to the developer.

PortForge closes that gap: paste in CUDA code, get back compiled, benchmarked, working HIP code.

## How It Works

1. **AST Parser Agent** — scans CUDA source files and identifies every CUDA-specific API call, kernel launch, and pattern that needs conversion, producing a structured JSON manifest.
2. **LLM Migration Agent** — sends the code + manifest to an AI coding model (via Fireworks AI, hosted on AMD hardware) to generate accurate HIP equivalents.
3. **Compilation + Error-Correction Loop** *(in progress)* — compiles generated HIP code with `hipcc` on AMD MI300X, automatically retries on errors.
4. **Benchmarking** *(planned)* — runs `rocprof` to measure real performance on AMD hardware.
5. **Frontend Dashboard** *(planned)* — visualizes the live migration, compilation status, and benchmark results.

## Tech Stack

- **AMD MI300X** — AMD Developer Cloud (compilation + benchmarking)
- **ROCm / HIP / hipcc / rocprof** — AMD's GPU compute toolchain
- **Fireworks AI API** — LLM inference (DeepSeek-V4-Pro)
- **Python** — core agent logic
- **Docker** — containerized submission (required by hackathon rules)

## Test Kernels

Five representative CUDA kernels selected from NVIDIA's official `cuda-samples` repository, ranging from simple to complex:

| Kernel | Complexity | Status |
|---|---|---|
| vectorAdd.cu | Simple | ✅ Migrated |
| matrixMul.cu | Simple/Medium | ✅ Migrated |
| convolutionSeparable.cu | Medium | ✅ Migrated |
| reduction_kernel.cu | Medium | ✅ Migrated |
| warpAggregatedAtomicsCG.cu | Hard (cooperative groups, atomics) | ✅ Migrated |

## Project Status

🚧 **In active development** — built for AMD Developer Hackathon: ACT II.

- [x] Project setup, environment, repo structure
- [x] AST Parser Agent
- [x] LLM Migration Agent (code generation, streaming, cost-optimized)
- [ ] hipcc compilation + error-correction loop (pending AMD Cloud access)
- [ ] rocprof benchmarking
- [ ] Frontend dashboard
- [ ] Docker containerization
- [ ] Demo polish

## Setup

*(Instructions to be finalized as the project nears submission.)*

```bash
git clone <repo-url>
cd portforge
pip install -r requirements.txt
cp .env.example .env  # add your Fireworks API key
python agents/run_parser_on_all.py
```

## Team

- **Muhammad Shoaib Altaf** — Lead
- *(Second teammate joining soon)*

## License

MIT
