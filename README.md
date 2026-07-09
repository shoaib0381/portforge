![PortForge](https://img.shields.io/badge/PortForge-AMD%20MI300X-CC0000?style=for-the-badge) ![ROCm](https://img.shields.io/badge/ROCm-7.2.4-ED1C24?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# PortForge

> AI-powered CUDA to AMD ROCm/HIP migration tool — automatically converts, compiles, and benchmarks GPU kernels on AMD MI300X using multi-agent AI pipeline.

**Built for:** AMD Developer Hackathon ACT II — Track 3 Unicorn

![PortForge Demo](docs/demo.gif)
<!-- Demo video coming soon -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/5e32dc14-2a87-43f4-a2bb-03de5de58bd5" width="80%" style="border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="PortForge Dashboard 1">
  <br>
  <img src="https://github.com/user-attachments/assets/6afb6b98-4017-4ae8-a0c8-50ddec558107" width="80%" style="border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="PortForge Dashboard 2">
  <br>
  <img src="https://github.com/user-attachments/assets/300d4fff-4f8c-481f-a3d4-c369deee90d5" width="80%" style="border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="PortForge Dashboard 3">
  <br>
  <img src="https://github.com/user-attachments/assets/bdcaec94-110d-47e7-867e-34d24b76fd63" width="80%" style="border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="PortForge Dashboard 4">
  <br>
  <img src="https://github.com/user-attachments/assets/e6eb455e-4eb5-4cfb-9cae-63887c9e197c" width="80%" style="border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="PortForge Dashboard 5">
</div>

## The Problem

For over 15 years, the GPU computing ecosystem has been heavily locked into NVIDIA's CUDA platform, amassing millions of developers and countless projects that rely exclusively on CUDA. As AMD brings incredibly powerful hardware like the MI300X to market, this software lock-in remains the largest barrier to entry.

Manually migrating a mature CUDA codebase to AMD's HIP/ROCm platform is an arduous process that can take weeks or months. Developers must painstakingly locate CUDA-specific APIs, memory management functions, and kernel launch parameters, replacing them with their HIP equivalents.

While AMD provides tools like `hipify-clang`, these traditional parsers typically only handle basic syntax translation. They lack the semantic understanding required to refactor complex logic, and they don't automatically compile and validate the generated code on target hardware. PortForge solves this end-to-end automatically.

## The Solution

PortForge is a fully automated, AI-driven migration pipeline that translates legacy CUDA code into highly optimized AMD HIP/ROCm code. It uses a custom AST parser to analyze the source, an intelligent LLM agent to semantically refactor the code, and then automatically connects to an AMD MI300X server to compile and benchmark the result.

```text
CUDA Code → [AST Parser] → [LLM Migration Agent] → [HIP Code] → [hipcc Compiler] → [rocprof Benchmark] → Results
```

## Key Features

- 🤖 **AI-powered semantic code understanding:** Goes beyond simple find/replace to actually comprehend and refactor complex kernel logic.
- ⚡ **Real AMD MI300X compilation validation:** Automatically connects via SSH to compile code on actual hardware.
- 📊 **rocprof performance benchmarking:** Provides real execution time profiling on the generated HIP kernels.
- 🔄 **Automatic error correction loop:** Intelligently analyzes compilation errors and refactors code until successful.
- 📁 **File upload support:** Easily drag and drop `.cu` files for instant translation.
- 🖥️ **Live code editor:** Interactive UI with dual-pane syntax highlighting and editing capabilities.
- 🌐 **REST API:** Fully featured endpoints for programmatic access and CI/CD integration.
- 💾 **Fallback caching:** Provides cached benchmark results when the GPU server is offline.

## Benchmark Results

| Kernel | Status | GPU Time | Conversions |
|--------|--------|----------|-------------|
| vectorAdd.cu | ✅ PASSED | 7,778 ns | 12 |
| matrixMul.cu | ✅ COMPILED | ~estimated | 15 |
| warpAggregatedAtomicsCG.cu | ✅ RAN | 1,485,581 ns | 31 |
| convolutionSeparable.cu | ⚠️ PARTIAL | - | 5 |
| reduction_kernel.cu | ⚠️ PARTIAL | - | 89 |

*Benchmarked on AMD MI300X (192GB HBM3) via ROCm 7.2.4*

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **GPU Hardware** | AMD MI300X (192GB HBM3) |
| **GPU Software** | ROCm 7.2.4, HIP, hipcc, rocprof |
| **AI Model** | Google Gemma 3 27B IT via Fireworks AI |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Code Analysis** | Custom AST Parser (regex + tree-sitter) |
| **Cloud** | AMD Developer Cloud (ATL1 region) |

## 🏆 Gemma 4 Bonus Challenge
PortForge participates in the Best Use of Gemma 4 bonus 
challenge ($6,000 prize pool). We use Google's Gemma model 
via Fireworks AI API as our core AI reasoning engine for 
CUDA-to-HIP code migration. The Gemma model analyzes CUDA 
kernel semantics, understands GPU programming patterns, and 
generates accurate HIP equivalents — demonstrating Gemma's 
capability in highly technical, domain-specific code 
intelligence tasks running on AMD infrastructure.

## Project Structure

```text
portforge/
├── agents/
│   ├── ast_parser.py          # CUDA AST analysis
│   ├── migration_agent.py     # LLM-powered HIP generation
│   └── test_fireworks_connection.py
├── backend/
│   └── api.py                 # FastAPI REST endpoints
├── frontend/
│   ├── index.html             # Main UI
│   ├── style.css              # Styling
│   ├── app.js                 # Frontend logic
│   └── assets/                # Images
├── kernels/
│   ├── raw/                   # Original CUDA files
│   └── converted/             # Migrated HIP files
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js (optional, for development)
- Fireworks AI API key
- AMD Developer Cloud account (for GPU compilation)

### Installation

```bash
# Clone the repository
git clone https://github.com/shoaib0381/portforge.git
cd portforge

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Add your FIREWORKS_API_KEY to .env

# Start backend
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8001 --reload

# Start frontend (new terminal)
cd frontend
powershell.exe -ExecutionPolicy Bypass -File "server.ps1"
# OR
python -m http.server 8000

# Open browser
http://localhost:8000
```

## API Documentation

- **GET** `/api/kernels` — List all kernels with stats
- **GET** `/api/kernel/{name}` — Get specific kernel data
- **POST** `/api/migrate` — Migrate custom CUDA code
- **POST** `/api/compile` — Compile on AMD MI300X
- **GET** `/api/status` — Pipeline status for all kernels

**Example request for migration:**
```bash
curl -X POST http://localhost:8001/api/migrate \
  -F "filename=custom_kernel.cu" \
  -F "cuda_code=$(cat custom_kernel.cu)"
```

## How It Works

1. **Upload or Select:** Upload a custom `.cu` file or select a legacy CUDA kernel template.
2. **AST Parsing:** The custom AST Parser scans the code for all CUDA-specific API calls, memory configurations, and headers.
3. **Agent Delegation:** The Migration Agent sends the source code and the AST manifest to the Gemma 3 27B IT model.
4. **Semantic Translation:** The LLM generates the equivalent HIP code alongside a detailed reasoning trace explaining its architectural choices.
5. **Hardware Compilation:** `hipcc` compiles the generated HIP code remotely on an AMD MI300X via SSH.
6. **Performance Benchmarking:** `rocprof` benchmarks the execution on real AMD hardware.
7. **Result Delivery:** Compilation status, logs, and execution times are streamed back to the interactive UI.

## 👥 Team

| | Name | Role |
|-|------|------|
| <img src="frontend/assets/shoaib.jpg" width="80" height="80" style="border-radius: 50%; object-fit: cover;"> | **Muhammad Shoaib Altaf** | Lead Developer |
| <img src="frontend/assets/faizan.jpg" width="80" height="80" style="border-radius: 50%; object-fit: cover;"> | **Faizan Haider** | AI Engineer |

## Hackathon Info

- **Event:** AMD Developer Hackathon ACT II
- **Track:** Track 3 — Unicorn (Open Innovation)
- **Platform:** lablab.ai
- **AMD Technology Used:** MI300X, ROCm 7.2.4, HIP, hipcc, rocprof

## License

MIT License — see LICENSE file

## Acknowledgments

- **AMD Developer Cloud** for MI300X access
- **Fireworks AI** for LLM inference API
- **NVIDIA cuda-samples** repository for test kernels
- **lablab.ai** for organizing the hackathon
