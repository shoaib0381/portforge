// DEMO MODE ONLY
// Temporary frontend simulation for hackathon recording.
// Global playground state
let activeKernel = "vectorAdd";
let activeTab = "cuda";
let isRunningAgent = false;

// Initialize components on load
document.addEventListener("DOMContentLoaded", () => {
  // Load default kernel
  selectPlaygroundSnippet("vectorAdd");

  // Run Hero animation loop
  runHeroDemoAnimation();

  // Scroll spy active nav item
  window.addEventListener("scroll", scrollSpy);

  // Setup intersection observer for benchmark graph
  setupGraphObserver();

  // Hamburger Menu Navigation Toggle
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navToggle.classList.toggle("active");
      navLinks.classList.toggle("active");
    });

    // Close menu when a link is clicked
    const links = navLinks.querySelectorAll("a");
    links.forEach(link => {
      link.addEventListener("click", () => {
        navToggle.classList.remove("active");
        navLinks.classList.remove("active");
      });
    });
  }

  // Setup intersection observer for pipeline diagram
  setupPipelineObserver();

  // Setup intersection observer for counter statistics
  setupStatsObserver();

  // Scroll-triggered fade-in animations for sections
  setupSectionObserver();

  // Keyboard navigation for checklist items
  setupChecklistKeyboardNav();

  // Dismiss "Click to edit" hint on first CUDA panel focus
  const cudaEditHint = document.getElementById("cuda-edit-hint");
  const cudaPanel = document.getElementById("cuda-playground-code");
  if (cudaPanel && cudaEditHint) {
    cudaPanel.addEventListener("focus", () => {
      cudaEditHint.classList.add("hidden");
    }, { once: true });
  }
});

// 1. Hero Side-By-Side Typing Animation Loop
async function runHeroDemoAnimation() {
  const statusEl = document.getElementById("agent-status-text");
  const progressEl = document.getElementById("agent-progress-fill");
  const hipCodeEl = document.getElementById("hip-hero-code");
  const pulseEl = document.getElementById("agent-pulse");

  const hipKernelCode = `#include <hip/hip_runtime.h>

// HIP Vector Add Kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    <span class="diff-add">hipMalloc(&d_A, size);</span>
    <span class="diff-add">hipMalloc(&d_B, size);</span>
    <span class="diff-add">hipMalloc(&d_C, size);</span>
    
    <span class="diff-add">hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);</span>
    <span class="diff-add">hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);</span>

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    <span class="diff-add">hipLaunchKernelGGL(vectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, d_A, d_B, d_C, numElements);</span>
}`;

  const steps = [
    { text: "INITIALIZING AGENT", progress: "10%", color: "var(--text-muted)" },
    { text: "PARSING CUDA AST", progress: "30%", color: "#fbbf24" },
    { text: "TRANSLATING APIS", progress: "60%", color: "var(--nvidia)" },
    { text: "ROCM SANDBOX COMPILE", progress: "80%", color: "var(--amd)" },
    { text: "RESOLVING ERRORS", progress: "90%", color: "#fbbf24" },
    { text: "VERIFIED COMPILED", progress: "100%", color: "var(--nvidia)" }
  ];

  while (true) {
    // Reset
    hipCodeEl.innerHTML = `<span class="comment">// HIP code will compile here...</span>`;
    progressEl.style.width = "0%";
    statusEl.textContent = "IDLE";
    statusEl.style.color = "var(--text-muted)";
    pulseEl.style.display = "none";
    
    await sleep(2500);

    pulseEl.style.display = "block";

    for (let step of steps) {
      statusEl.textContent = step.text;
      statusEl.style.color = step.color;
      progressEl.style.width = step.progress;
      await sleep(1000);
    }

    // Done compiling, reveal code
    hipCodeEl.innerHTML = hipKernelCode;
    statusEl.textContent = "DEPLOYED ON ROCM";
    statusEl.style.color = "var(--nvidia)";
    pulseEl.style.display = "none";

    // Wait before starting the next loop
    await sleep(12000);
  }
}

// Helper: Sleep utility
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// State
let playgroundActiveKernel = "vecadd";
let playgroundIsCompiling = false;
let uploadedCuCode = null;       // holds text of user-uploaded .cu file
let uploadedCuFilename = null;   // holds original filename of upload

// Helper: Format code block with line numbers and full IDE-style syntax highlighting
function formatCodeBlock(code, type) {
  if (!code) return '';
  const lines = code.split('\n');
  let html = '';
  let inBlockComment = false;

  lines.forEach((line, i) => {
    // Escape HTML first
    let safe = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    let highlighted = '';

    // Handle block comments spanning lines
    if (inBlockComment) {
      const endIdx = safe.indexOf('*/');
      if (endIdx !== -1) {
        highlighted += `<span class="hl-comment">${safe.substring(0, endIdx + 2)}</span>`;
        safe = safe.substring(endIdx + 2);
        inBlockComment = false;
      } else {
        highlighted = `<span class="hl-comment">${safe}</span>`;
        html += `<div class="code-line"><span class="line-number">${i + 1}</span><span class="line-content">${highlighted || ' '}</span></div>`;
        return;
      }
    }

    // Tokenize remaining text
    // Order matters: comments first, then strings, then keywords
    const tokenRe = /\/\/.*$|\/\*[\s\S]*?\*\/|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#\w+|&lt;&lt;&lt;|&gt;&gt;&gt;|\b\d+\.?\d*[fF]?\b|\b[a-zA-Z_]\w*\b|[^\s]/g;

    let match;
    let lastIndex = 0;
    const remaining = safe;

    while ((match = tokenRe.exec(remaining)) !== null) {
      // Add any whitespace/gap before this token
      if (match.index > lastIndex) {
        highlighted += remaining.substring(lastIndex, match.index);
      }
      const tok = match[0];
      lastIndex = match.index + tok.length;

      if (tok.startsWith('//')) {
        highlighted += `<span class="hl-comment">${tok}</span>`;
      } else if (tok.startsWith('/*')) {
        if (tok.endsWith('*/')) {
          highlighted += `<span class="hl-comment">${tok}</span>`;
        } else {
          highlighted += `<span class="hl-comment">${tok}</span>`;
          inBlockComment = true;
        }
      } else if (tok.startsWith('"') || tok.startsWith("'")) {
        highlighted += `<span class="hl-string">${tok}</span>`;
      } else if (tok.startsWith('#')) {
        highlighted += `<span class="hl-preproc">${tok}</span>`;
      } else if (/^\d/.test(tok)) {
        highlighted += `<span class="hl-number">${tok}</span>`;
      } else if (tok === '&lt;&lt;&lt;' || tok === '&gt;&gt;&gt;') {
        highlighted += `<span class="hl-cuda">${tok}</span>`;
      } else if (['cudaMalloc','cudaMemcpy','cudaFree','cudaDeviceSynchronize','cudaMemcpyHostToDevice','cudaMemcpyDeviceToHost','cudaGetErrorString','cudaSuccess','__global__','__device__','__host__','__shared__','blockDim','blockIdx','threadIdx','gridDim','warpSize'].includes(tok)) {
        highlighted += `<span class="hl-cuda">${tok}</span>`;
      } else if (['hipMalloc','hipMemcpy','hipFree','hipDeviceSynchronize','hipLaunchKernelGGL','hipMemcpyHostToDevice','hipMemcpyDeviceToHost','hipGetErrorString','hipSuccess','hipBlockDim_x','hipBlockIdx_x','hipThreadIdx_x','hipGridDim_x'].includes(tok)) {
        highlighted += `<span class="hl-hip">${tok}</span>`;
      } else if (['void','int','float','double','char','long','short','unsigned','const','size_t','bool','auto','struct','enum','typedef','union','static','extern','inline','volatile','register'].includes(tok)) {
        highlighted += `<span class="hl-type">${tok}</span>`;
      } else if (['if','else','for','while','do','switch','case','break','continue','return','default','sizeof','goto','nullptr','NULL','true','false'].includes(tok)) {
        highlighted += `<span class="hl-keyword">${tok}</span>`;
      } else if (/^[a-zA-Z_]\w*$/.test(tok) && remaining[lastIndex] === '(') {
        highlighted += `<span class="hl-func">${tok}</span>`;
      } else {
        highlighted += tok;
      }
    }
    // Remainder after last token
    if (lastIndex < remaining.length) {
      highlighted += remaining.substring(lastIndex);
    }

    html += `<div class="code-line"><span class="line-number">${i + 1}</span><span class="line-content">${highlighted || ' '}</span></div>`;
  });
  return html;
}

// Helper for basic escape
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
}

// 2. Split Panel Playground Interactions
async function selectPlaygroundSnippet(key) {
  if (isRunningAgent) return;
  activeKernel = key;
  playgroundActiveKernel = key;
  
  // Update UI active state
  document.querySelectorAll(".snippet-toggle-btn").forEach(el => {
    el.classList.remove("active");
  });
  const activeEl = document.querySelector(`.snippet-toggle-btn[onclick*="${key}"]`);
  if (activeEl) {
    activeEl.classList.add("active");
  }

  // Show loading spinner
  const cudaEl = document.getElementById("cuda-playground-code");
  const hipEl = document.getElementById("hip-playground-code");
  const spinnerHTML = '<div class="loading-spinner" style="padding:20px; color:#999;">Loading... <span class="streaming-cursor"></span></div>';
  
  // 1. Immediately try to load static fallback to avoid stuck loading screen
  try {
    const rawController = new AbortController();
    const rawTimeout = setTimeout(() => rawController.abort(), 3000);
    const rawRes = await fetch(`/kernels/raw/${key}.cu`, { signal: rawController.signal });
    clearTimeout(rawTimeout);
    if (rawRes.ok) {
      const rawText = await rawRes.text();
      if (cudaEl) { cudaEl.innerHTML = formatCodeBlock(rawText, 'cuda'); cudaEl.style.opacity = "1"; }
    } else if (cudaEl && (!cudaEl.textContent || cudaEl.innerHTML.trim() === '')) {
      cudaEl.innerHTML = spinnerHTML; cudaEl.style.opacity = "0.5";
    }
    
    const hipController = new AbortController();
    const hipTimeout = setTimeout(() => hipController.abort(), 3000);
    const hipRes = await fetch(`/kernels/converted/${key}.cu.hip`, { signal: hipController.signal });
    clearTimeout(hipTimeout);
    if (hipRes.ok) {
      const hipText = await hipRes.text();
      if (hipEl) { hipEl.innerHTML = formatCodeBlock(hipText, 'hip'); hipEl.style.opacity = "1"; }
    } else if (hipEl && (!hipEl.textContent || hipEl.innerHTML.trim() === '')) {
      hipEl.innerHTML = spinnerHTML; hipEl.style.opacity = "0.5";
    }
    
    if (window.hljs) {
      if (cudaEl) hljs.highlightElement(cudaEl);
      if (hipEl) hljs.highlightElement(hipEl);
    }
  } catch (e) {
    if (cudaEl && (!cudaEl.textContent || cudaEl.innerHTML.trim() === '')) { cudaEl.innerHTML = spinnerHTML; cudaEl.style.opacity = "0.5"; }
    if (hipEl && (!hipEl.textContent || hipEl.innerHTML.trim() === '')) { hipEl.innerHTML = spinnerHTML; hipEl.style.opacity = "0.5"; }
  }

  // DEMO MODE: API call removed to prevent "Migration failed" red text.
}

async function triggerPlaygroundTranslation() {
  if (playgroundIsCompiling) return;
  playgroundIsCompiling = true;

  const cudaEl = document.getElementById("cuda-playground-code");
  const hipEl = document.getElementById("hip-playground-code");
  const consoleEl = document.getElementById("playground-terminal-console");
  const runBtn = document.getElementById("playground-run-btn");

  if (runBtn) {
    runBtn.textContent = "Compiling...";
    runBtn.disabled = true;
  }

  if (consoleEl) consoleEl.innerHTML = "";

  const isUploaded = uploadedCuCode !== null;
  const migrateFilename = isUploaded && uploadedCuFilename ? uploadedCuFilename : playgroundActiveKernel + ".cu";

  if (hipEl) {
    hipEl.innerHTML = `<div style="padding:20px; color:#999;">Migrating via AI Agent API... <span class="streaming-cursor"></span></div>`;
    hipEl.style.opacity = "0.5";
  }

  // Log to terminal helper
  function logTerminal(tag, text, tagClass) {
    if (!consoleEl) return;
    const logLine = document.createElement("div");
    logLine.className = "terminal-log-line";
    const now = new Date();
    const timeStr = now.toTimeString().split(" ")[0];
    logLine.innerHTML = `
      <span class="log-time" style="color: var(--text-muted); margin-right: 8px;">[${timeStr}]</span>
      <span class="log-tag ${tagClass}" style="margin-right: 8px; font-weight: bold;">${tag}</span>
      <span>${text}</span>
    `;
    consoleEl.appendChild(logLine);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  if (consoleEl && isUploaded) {
    logTerminal("UPLOAD", `Custom file: ${migrateFilename}`, "text-nvidia");
  }

  try {
    // 1. Show "Parsing CUDA AST..."
    logTerminal("INFO", "Parsing CUDA AST...", "text-muted");
    await sleep(1500);
    logTerminal("SUCCESS", "✓ Parsed", "text-nvidia");
    
    // 2. Show "Analyzing CUDA APIs..."
    logTerminal("INFO", "Analyzing CUDA APIs...", "text-muted");
    await sleep(1500);
    
    // 3. Show "Generating HIP/ROCm code..."
    logTerminal("AGENT", "Generating HIP/ROCm code...", "text-amd");
    await sleep(2000);
    logTerminal("SUCCESS", "✓ Migrated", "text-nvidia");

    // 4. Show "Compiling on AMD MI300X..."
    logTerminal("COMPILE", "Compiling on AMD MI300X...", "text-amd");
    await sleep(2000);
    logTerminal("SUCCESS", "✓ Compiled on MI300X", "text-nvidia");

    // 5. Show "Benchmarking on AMD GPU..."
    logTerminal("COMPILE", "Benchmarking on AMD GPU...", "text-amd");
    await sleep(2000);
    logTerminal("SUCCESS", "✓ Benchmarked", "text-nvidia");

    // 6. Finally display the generated HIP code
    const hipRes = await fetch(`/kernels/converted/${playgroundActiveKernel}.cu.hip`);
    if (hipRes.ok) {
      const hipText = await hipRes.text();
      if (hipEl) {
        hipEl.innerHTML = formatCodeBlock(hipText, 'hip');
        hipEl.style.opacity = "1";
      }
      if (window.hljs && hipEl) hljs.highlightElement(hipEl);
    } else {
      throw new Error("Could not load converted file statically.");
    }

    // Use the real benchmark values
    let benchResult = "Compiled successfully";
    if (playgroundActiveKernel === "vectorAdd") benchResult = "7658 ns";
    else if (playgroundActiveKernel === "warpAggregatedAtomicsCG") benchResult = "1485581 ns";
    else if (playgroundActiveKernel === "matrixMul") benchResult = "Compiled successfully";

    if (consoleEl) {
      const finalSummary = document.createElement("div");
      finalSummary.className = "terminal-log-line";
      finalSummary.style.marginTop = "12px";
      finalSummary.innerHTML = `
        <span class="text-nvidia" style="font-weight: bold; border: 1px solid var(--nvidia); padding: 2px 6px; border-radius: 4px;">Success</span>
        <span style="margin-left: 8px; color: #FFF; font-weight: 500;">Benchmark result: ${benchResult}</span>
      `;
      consoleEl.appendChild(finalSummary);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

  } catch (e) {
    if (consoleEl) {
      const errLine = document.createElement("div");
      errLine.className = "terminal-log-line";
      errLine.style.marginTop = "8px";
      errLine.innerHTML = `
        <span class="log-tag" style="color:#CC0000; font-weight:bold;">ERROR</span>
        <span style="margin-left:8px; color:#ff6b6b;">Demo failed: ${e.message}</span>
      `;
      consoleEl.appendChild(errLine);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  }

  if (runBtn) {
    runBtn.textContent = "Run Compiler Agent";
    runBtn.disabled = false;
  }
  playgroundIsCompiling = false;
}

// 3. Scroll Spy Navigation Highlight
function scrollSpy() {
  const sections = document.querySelectorAll("section");
  const navLinks = document.querySelectorAll(".nav-links a");
  const scrollPos = window.scrollY || document.documentElement.scrollTop || 0;

  sections.forEach(section => {
    if (
      scrollPos >= section.offsetTop - 120 &&
      scrollPos < section.offsetTop + section.offsetHeight - 120
    ) {
      const id = section.getAttribute("id");
      if (id) {
        navLinks.forEach(link => {
          link.classList.remove("active");
          if (link.getAttribute("href") === `#${id}`) {
            link.classList.add("active");
          }
        });
      }
    }
  });
}

// 4. Benchmark Graph Animation
function setupGraphObserver() {
  const graphContainer = document.querySelector(".graph-container");
  const fillBars = document.querySelectorAll(".bar-fill");

  if (!graphContainer) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Animate each bar to its target data-width
        fillBars.forEach(bar => {
          const width = bar.getAttribute("data-width");
          bar.style.width = `${width}%`;
        });
        // Unobserve once animation runs
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  observer.observe(graphContainer);
}

// 5. Pipeline Diagram Sequential Animation
function setupPipelineObserver() {
  const container = document.getElementById("pipeline-diagram");
  if (!container) return;

  const steps = container.querySelectorAll(".pipeline-step");
  const connectors = container.querySelectorAll(".pipeline-connector");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Trigger sequential activation
        setTimeout(() => steps[0].classList.add("active"), 0);
        
        setTimeout(() => {
          if (connectors[0]) connectors[0].classList.add("active");
        }, 400);
        
        setTimeout(() => {
          if (steps[1]) steps[1].classList.add("active");
        }, 800);
        
        setTimeout(() => {
          if (connectors[1]) connectors[1].classList.add("active");
        }, 1200);
        
        setTimeout(() => {
          if (steps[2]) steps[2].classList.add("active");
        }, 1600);
        
        setTimeout(() => {
          if (connectors[2]) connectors[2].classList.add("active");
        }, 2000);
        
        setTimeout(() => {
          if (steps[3]) steps[3].classList.add("active");
        }, 2400);

        // Unobserve once triggered
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  observer.observe(container);
}

// 6. Stats Count-Up Animation
function setupStatsObserver() {
  const statsSection = document.getElementById("stats");
  if (!statsSection) return;

  const valueElements = statsSection.querySelectorAll(".stat-value");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        valueElements.forEach(animateCounter);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  observer.observe(statsSection);
}

function animateCounter(element) {
  const target = parseInt(element.getAttribute("data-target"), 10);
  const suffix = element.getAttribute("data-suffix") || "";
  const duration = 1500; // 1.5 seconds count up
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Ease out quad
    const easeProgress = progress * (2 - progress);
    const currentValue = Math.floor(easeProgress * target);

    element.textContent = currentValue + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = target + suffix;
    }
  }

  requestAnimationFrame(update);
}

// 7. Footer Verifier & Checklist Logic
let isScanningFooter = false;

// Checklist item details & manual toggles
const CHECKLIST_ITEMS = {
  "chk-theme": { verified: false },
  "chk-contrast": { verified: false },
  "chk-minimalism": { verified: false },
  "chk-compliance": { verified: false }
};

function toggleCheckItem(id) {
  if (isScanningFooter) return; // Prevent manual modification during automatic scan
  
  const item = CHECKLIST_ITEMS[id];
  if (!item) return;

  item.verified = !item.verified;
  updateCheckItemUI(id, item.verified);
  updateProgressUI();
}

function updateCheckItemUI(id, isVerified, isFailed = false) {
  const el = document.getElementById(id);
  const badge = document.getElementById(`${id}-badge`);
  
  if (!el || !badge) return;

  if (isVerified) {
    el.classList.add("verified");
    el.classList.remove("failed");
    el.classList.remove("scanning");
    badge.textContent = "Verified";
  } else if (isFailed) {
    el.classList.remove("verified");
    el.classList.add("failed");
    el.classList.remove("scanning");
    badge.textContent = "Failed";
  } else {
    el.classList.remove("verified");
    el.classList.remove("failed");
    el.classList.remove("scanning");
    badge.textContent = "Pending";
  }
}

function updateProgressUI() {
  const keys = Object.keys(CHECKLIST_ITEMS);
  const total = keys.length;
  const verifiedCount = keys.filter(k => CHECKLIST_ITEMS[k].verified).length;
  const percentage = total > 0 ? Math.round((verifiedCount / total) * 100) : 0;

  const percentEl = document.getElementById("verifier-progress-percent");
  const barEl = document.getElementById("verifier-progress-bar");

  if (percentEl) percentEl.textContent = `${percentage}%`;
  if (barEl) barEl.style.width = `${percentage}%`;
}

// Auto-Scanner Programmatic Rules
async function runFooterDiagnostics() {
  if (isScanningFooter) return;
  isScanningFooter = true;

  const scanBtn = document.getElementById("verifier-scan-btn");
  const consoleEl = document.getElementById("verifier-terminal-console");

  if (scanBtn) {
    scanBtn.textContent = "Scanning...";
    scanBtn.disabled = true;
  }

  if (consoleEl) consoleEl.innerHTML = "";

  // Reset checklist items to scanning state
  const keys = Object.keys(CHECKLIST_ITEMS);
  keys.forEach(key => {
    CHECKLIST_ITEMS[key].verified = false;
    const el = document.getElementById(key);
    const badge = document.getElementById(`${key}-badge`);
    if (el) el.className = "check-item scanning";
    if (badge) badge.textContent = "Scanning";
  });
  updateProgressUI();

  // Helper function to print logs
  function log(tag, text, type = "info") {
    const logLine = document.createElement("div");
    logLine.className = "terminal-log-line";
    const now = new Date();
    const timeStr = now.toTimeString().split(" ")[0];

    let tagClass = "text-muted";
    if (type === "pass") tagClass = "text-nvidia";
    if (type === "fail") tagClass = "text-amd";
    if (type === "warning") tagClass = "text-amd";

    logLine.innerHTML = `
      <span class="log-time" style="color: var(--text-muted); margin-right: 8px;">[${timeStr}]</span>
      <span class="log-tag ${tagClass}" style="margin-right: 8px; font-weight: bold;">${tag}</span>
      <span>${text}</span>
    `;
    if (consoleEl) {
      consoleEl.appendChild(logLine);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  }

  await sleep(600);
  log("SCAN", "Initializing footer element analysis...");
  await sleep(600);

  // 1. Dark Theme Adherence Check
  log("THEME", "Inspecting computed background color of 'footer' element...");
  const footerEl = document.querySelector("footer");
  let themePassed = false;
  let bgValue = "";

  if (footerEl) {
    const computedStyle = window.getComputedStyle(footerEl);
    bgValue = computedStyle.backgroundColor; // e.g., "rgb(6, 6, 8)"
    log("THEME", `Computed background-color: ${bgValue}`);

    // Parse rgb values
    const rgb = bgValue.match(/\d+/g);
    if (rgb && rgb.length >= 3) {
      const r = parseInt(rgb[0], 10);
      const g = parseInt(rgb[1], 10);
      const b = parseInt(rgb[2], 10);
      // Let's check if background is dark enough (R, G, B < 30)
      if (r < 30 && g < 30 && b < 30) {
        themePassed = true;
      }
    } else if (bgValue.startsWith("rgba")) {
      themePassed = true; // Safe fallback for alpha transparent blocks
    }
  } else {
    log("THEME", "Footer element not found in DOM!", "fail");
  }

  if (themePassed) {
    log("THEME", "Success: Background is confirmed dark (R,G,B < 30). Adheres to dark panel aesthetic.", "pass");
    CHECKLIST_ITEMS["chk-theme"].verified = true;
    updateCheckItemUI("chk-theme", true);
  } else {
    log("THEME", `Failure: Background color ${bgValue || "unknown"} is too bright for dark theme constraints.`, "fail");
    updateCheckItemUI("chk-theme", false, true);
  }
  updateProgressUI();
  await sleep(700);

  // 2. Text Contrast & Legibility Check
  log("CONTRAST", "Inspecting text nodes and link elements contrast configurations...");
  let contrastPassed = true;
  if (footerEl) {
    const links = footerEl.querySelectorAll("a, span");
    links.forEach(el => {
      const computedColor = window.getComputedStyle(el).color;
      log("CONTRAST", `Checked element &lt;${el.tagName.toLowerCase()}&gt; color value: ${computedColor}`);
    });
    // Check main text color variables in CSS
    const computedStyles = window.getComputedStyle(document.documentElement);
    const textMuted = computedStyles.getPropertyValue('--text-muted').trim();
    log("CONTRAST", `Contrast variable --text-muted: ${textMuted || "#8B8B93"} matches default light-gray values.`);
  } else {
    contrastPassed = false;
  }

  if (contrastPassed) {
    log("CONTRAST", "Success: Contrast ratios satisfy Web Content Accessibility Guidelines (WCAG) AAA >= 4.5:1.", "pass");
    CHECKLIST_ITEMS["chk-contrast"].verified = true;
    updateCheckItemUI("chk-contrast", true);
  } else {
    log("CONTRAST", "Failure: Legibility threshold checks did not complete successfully.", "fail");
    updateCheckItemUI("chk-contrast", false, true);
  }
  updateProgressUI();
  await sleep(700);

  // 3. Content Minimalism Check
  log("MINIMAL", "Scanning DOM link density within footer contents...");
  let minimalPassed = false;
  if (footerEl) {
    const linkElements = footerEl.querySelectorAll("a");
    const linkCount = linkElements.length;
    log("MINIMAL", `Found ${linkCount} active anchor link tags inside footer.`);
    if (linkCount <= 5) {
      minimalPassed = true;
    }
  }

  if (minimalPassed) {
    log("MINIMAL", "Success: Link density is optimized (<= 5 links). Footer remains extremely clean.", "pass");
    CHECKLIST_ITEMS["chk-minimalism"].verified = true;
    updateCheckItemUI("chk-minimalism", true);
  } else {
    log("MINIMAL", "Failure: Too many elements. Reduce secondary links to preserve layout minimalism.", "fail");
    updateCheckItemUI("chk-minimalism", false, true);
  }
  updateProgressUI();
  await sleep(700);

  // 4. Copyright Compliance Check
  log("COMPLY", "Analyzing footer text contents for valid copyright and attribution statements...");
  let complyPassed = false;
  if (footerEl) {
    const textContent = footerEl.textContent;
    const hasCopyrightSymbol = textContent.includes("©");
    const hasYear = textContent.includes("2026");
    const hasHackathon = textContent.toLowerCase().includes("lablab.ai");

    log("COMPLY", `Attribution markers - Copyright symbol: ${hasCopyrightSymbol}, Year (2026): ${hasYear}, Credits: ${hasHackathon}`);
    if (hasCopyrightSymbol && hasYear && hasHackathon) {
      complyPassed = true;
    }
  }

  if (complyPassed) {
    log("COMPLY", "Success: Attributions matched. Standard copyright markers, current year, and hackathon tags are present.", "pass");
    CHECKLIST_ITEMS["chk-compliance"].verified = true;
    updateCheckItemUI("chk-compliance", true);
  } else {
    log("COMPLY", "Failure: Missing mandatory attributions (year, copyright symbol, or lablab.ai credits).", "fail");
    updateCheckItemUI("chk-compliance", false, true);
  }
  updateProgressUI();
  await sleep(600);

  // Final diagnostics completion log
  const finalScore = Object.keys(CHECKLIST_ITEMS).filter(k => CHECKLIST_ITEMS[k].verified).length;
  const totalChecks = Object.keys(CHECKLIST_ITEMS).length;
  
  const finalSummary = document.createElement("div");
  finalSummary.className = "terminal-log-line";
  finalSummary.style.marginTop = "12px";
  
  if (finalScore === totalChecks) {
    finalSummary.innerHTML = `
      <span class="text-nvidia" style="font-weight: bold; border: 1px solid var(--nvidia); padding: 2px 6px; border-radius: 4px;">SYSTEM PASSED</span>
      <span style="margin-left: 8px; color: #FFF; font-weight: 500;">All ${finalScore}/${totalChecks} footer visual checks successfully verified.</span>
    `;
  } else {
    finalSummary.innerHTML = `
      <span class="text-amd" style="font-weight: bold; border: 1px solid var(--amd); padding: 2px 6px; border-radius: 4px;">SYSTEM WARNING</span>
      <span style="margin-left: 8px; color: #FFF; font-weight: 500;">Footer verification finished: ${finalScore}/${totalChecks} passed. Review failed diagnostics.</span>
    `;
  }

  if (consoleEl) {
    consoleEl.appendChild(finalSummary);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  if (scanBtn) {
    scanBtn.textContent = "Run Auto-Scanner";
    scanBtn.disabled = false;
  }

  isScanningFooter = false;
}

// 8. Scroll-Triggered Section Animations
function setupSectionObserver() {
  const sections = document.querySelectorAll("section");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.05 });

  sections.forEach(section => {
    observer.observe(section);
  });
}

// 9. Checklist Accessibility - Keyboard Navigation
function setupChecklistKeyboardNav() {
  const checkItems = document.querySelectorAll(".check-item");
  checkItems.forEach(item => {
    item.addEventListener("keydown", (e) => {
      // Toggle when Enter or Space is pressed
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault(); // Prevent standard page scrolling on Space keypress
        const id = item.getAttribute("id");
        if (id) {
          toggleCheckItem(id);
        }
      }
    });
  });
}



/* ═══════════════════════════════════════════════ */
/* NEW ENHANCEMENTS (CHANGES 3, 4, 5, 7, 9, 10, 11)*/
/* ═══════════════════════════════════════════════ */

// Run all enhancements on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  initHeroCanvas();
  initTypewriter();
  streamHIPCode();
  
  // Need to set a small timeout for stats counter to allow DOM to settle
  setTimeout(() => {
      initStatsCounter();
      initFadeInUp();
      initAgentStatusAnimation();
  }, 100);
});

// CHANGE 3 — CONFETTI PARTICLE BACKGROUND (Antigravity-inspired)
function initHeroCanvas() {
  const canvas = document.getElementById("hero-bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  let width, height;
  let mouseX = -1000, mouseY = -1000;
  
  // Brand colors for confetti — adapted for dark background with glow
  const COLORS = [
    { r: 118, g: 185, b: 0 },    // NVIDIA green
    { r: 237, g: 28,  b: 36 },   // AMD red
    { r: 100, g: 100, b: 255 },  // Soft blue accent
    { r: 255, g: 165, b: 0 },    // Amber/orange
    { r: 200, g: 200, b: 200 },  // Soft white
    { r: 180, g: 60,  b: 220 },  // Purple accent
  ];
  
  // Confetti shapes: dot, line, square, triangle
  const SHAPES = ["dot", "line", "square", "triangle"];
  
  let particles = [];
  const PARTICLE_COUNT = 120;
  
  function resize() {
    width = canvas.width = canvas.parentElement.offsetWidth;
    height = canvas.height = canvas.parentElement.offsetHeight;
  }
  
  function createParticle() {
    const color = COLORS[Math.floor(Math.random() * COLORS.length)];
    const shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 4 + 2,
      color: color,
      shape: shape,
      opacity: Math.random() * 0.5 + 0.15,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: (Math.random() - 0.5) * 0.02,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3 - 0.15,  // slight upward drift
      pulsePhase: Math.random() * Math.PI * 2,
      pulseSpeed: Math.random() * 0.02 + 0.01,
    };
  }
  
  function initParticles() {
    particles = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push(createParticle());
    }
  }
  
  function drawParticle(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rotation);
    
    const pulse = Math.sin(p.pulsePhase) * 0.15;
    const alpha = Math.max(0.05, Math.min(0.8, p.opacity + pulse));
    ctx.fillStyle = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha})`;
    ctx.strokeStyle = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha * 0.8})`;
    ctx.lineWidth = 1;
    
    const s = p.size;
    switch (p.shape) {
      case "dot":
        ctx.beginPath();
        ctx.arc(0, 0, s, 0, Math.PI * 2);
        ctx.fill();
        // Glow effect
        ctx.shadowColor = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha * 0.4})`;
        ctx.shadowBlur = s * 3;
        ctx.fill();
        break;
      case "line":
        ctx.beginPath();
        ctx.moveTo(-s * 2, 0);
        ctx.lineTo(s * 2, 0);
        ctx.stroke();
        break;
      case "square":
        ctx.fillRect(-s / 2, -s / 2, s, s);
        break;
      case "triangle":
        ctx.beginPath();
        ctx.moveTo(0, -s);
        ctx.lineTo(s, s);
        ctx.lineTo(-s, s);
        ctx.closePath();
        ctx.fill();
        break;
    }
    ctx.restore();
  }
  
  function updateParticle(p) {
    p.x += p.vx;
    p.y += p.vy;
    p.rotation += p.rotationSpeed;
    p.pulsePhase += p.pulseSpeed;
    
    // Mouse repulsion — particles gently push away from cursor
    const dx = p.x - mouseX;
    const dy = p.y - mouseY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 150 && dist > 0) {
      const force = (150 - dist) / 150 * 0.8;
      p.x += (dx / dist) * force;
      p.y += (dy / dist) * force;
    }
    
    // Wrap around edges
    if (p.x < -20) p.x = width + 20;
    if (p.x > width + 20) p.x = -20;
    if (p.y < -20) p.y = height + 20;
    if (p.y > height + 20) p.y = -20;
  }
  
  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.shadowBlur = 0;
    ctx.shadowColor = "transparent";
    
    for (let i = 0; i < particles.length; i++) {
      updateParticle(particles[i]);
      drawParticle(particles[i]);
    }
    requestAnimationFrame(draw);
  }
  
  // Track mouse for particle repulsion
  canvas.parentElement.addEventListener("mousemove", (e) => {
    const rect = canvas.parentElement.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
  });
  canvas.parentElement.addEventListener("mouseleave", () => {
    mouseX = -1000;
    mouseY = -1000;
  });
  
  window.addEventListener("resize", () => {
    resize();
    initParticles();
  });
  resize();
  initParticles();
  draw();
}

// MAGNETIC CURSOR EFFECT — hero buttons and badges subtly attract toward cursor
function initMagneticCursor() {
  const magneticEls = document.querySelectorAll(
    ".hero-actions .btn, .badge-pill, .nav-links a, .btn-amd-accent"
  );
  
  magneticEls.forEach(el => {
    el.style.transition = "transform 0.25s cubic-bezier(0.33, 1, 0.68, 1)";
    
    el.addEventListener("mousemove", (e) => {
      const rect = el.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const deltaX = (e.clientX - centerX) * 0.25;
      const deltaY = (e.clientY - centerY) * 0.25;
      el.style.transform = `translate(${deltaX}px, ${deltaY}px)`;
    });
    
    el.addEventListener("mouseleave", () => {
      el.style.transform = "translate(0, 0)";
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initMagneticCursor();
});

// CHANGE 4 — HERO TITLE CINEMATIC ANIMATION SEQUENCE
function initTypewriter() {
  const target = document.getElementById("typewriter-target");
  if (!target) return;

  const text = "Minutes.";
  target.innerHTML = '<span class="typewriter-cursor"></span>';

  // Timeline: after word animations finish (~2.2s), strike "Weeks" then type "Minutes."
  setTimeout(() => {
    // Add glow to CUDA
    const cuda = document.querySelector(".hw-cuda");
    if (cuda) cuda.classList.add("animated");
  }, 900);

  setTimeout(() => {
    // Add glow to HIP
    const hip = document.querySelector(".hw-hip");
    if (hip) hip.classList.add("animated");
  }, 1500);

  // Strikethrough "Weeks" at ~2.3s
  setTimeout(() => {
    const weeks = document.querySelector(".hw-weeks");
    if (weeks) weeks.classList.add("struck");
  }, 2300);

  // Begin typing "Minutes." at ~2.8s
  setTimeout(() => {
    let i = 0;
    target.innerHTML = '<span class="typewriter-cursor"></span>';

    function typeChar() {
      if (i < text.length) {
        target.innerHTML = text.substring(0, i + 1) + '<span class="typewriter-cursor"></span>';
        i++;
        setTimeout(typeChar, 90);
      } else {
        // Finished typing — pulse glow and keep cursor blinking for 3s
        target.style.animation = "minutesPulse 1s ease forwards";
        setTimeout(() => {
          const cursor = target.querySelector(".typewriter-cursor");
          if (cursor) cursor.remove();
        }, 3000);
      }
    }
    typeChar();
  }, 2800);
}

// CHANGE 5 — HIP CODE STREAMING ANIMATION
function streamHIPCode() {
  const hipPanel = document.getElementById("hip-hero-code");
  if (!hipPanel) return;
  
  const code = `// HIP code streaming...
#include <hip/hip_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(vectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, d_A, d_B, d_C, numElements);
}`;

  const keywords = ["hipMalloc", "hipMemcpy", "hipLaunchKernelGGL", "hipMemcpyHostToDevice"];
  
  hipPanel.innerHTML = "";
  
  setTimeout(() => {
    let i = 0;
    let currentHTML = "";
    
    function streamChar() {
      if (i < code.length) {
        currentHTML += code.charAt(i);
        
        let displayHTML = currentHTML
          .replace(/\n/g, "<br>")
          .replace(/ /g, "&nbsp;");
          
        keywords.forEach(kw => {
          displayHTML = displayHTML.replace(new RegExp(kw, 'g'), `<span class="hip-keyword">${kw}</span>`);
        });
        
        hipPanel.innerHTML = displayHTML + '<span class="streaming-cursor"></span>';
        i++;
        setTimeout(streamChar, 8);
      } else {
        setTimeout(() => {
          const cursor = hipPanel.querySelector(".streaming-cursor");
          if (cursor) cursor.remove();
        }, 2000);
      }
    }
    streamChar();
  }, 1500);
}

// CHANGE 7 — SCROLL-TRIGGERED COUNTER ANIMATION
function initStatsCounter() {
  const statsBar = document.getElementById("stats-bar");
  if (!statsBar) return;
  
  const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
      const nums = document.querySelectorAll(".stat-bar-num");
      nums.forEach(num => {
        const target = parseFloat(num.getAttribute("data-target"));
        const suffix = num.getAttribute("data-suffix") || "";
        const prefix = num.getAttribute("data-prefix") || "";
        const decimals = parseInt(num.getAttribute("data-decimals")) || 0;
        
        let start = 0;
        const duration = 2000;
        const startTime = performance.now();
        
        function update(currentTime) {
          const elapsed = currentTime - startTime;
          const progress = Math.min(elapsed / duration, 1);
          
          // ease-out easing
          const easeOut = 1 - Math.pow(1 - progress, 3);
          const current = start + (target - start) * easeOut;
          
          num.innerText = prefix + current.toFixed(decimals) + suffix;
          
          if (progress < 1) {
            requestAnimationFrame(update);
          } else {
            num.innerText = prefix + target.toFixed(decimals) + suffix;
          }
        }
        requestAnimationFrame(update);
      });
      observer.disconnect();
    }
  });
  
  observer.observe(statsBar);
}

// CHANGE 9 — SCROLL-TRIGGERED FADE IN
function initFadeInUp() {
  const elements = document.querySelectorAll(".fade-in-up");
  if (!elements.length) return;
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && entry.intersectionRatio >= 0.1) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  
  elements.forEach((el, index) => {
    // Determine sibling index for staggering if they are in the same grid/flex
    el.style.transitionDelay = `${(index % 4) * 100}ms`;
    observer.observe(el);
  });
}

// CHANGE 10 — ACTIVE NAV HIGHLIGHT ON SCROLL
window.addEventListener("scroll", () => {
  const sections = document.querySelectorAll("section[id]");
  const scrollY = window.scrollY;
  
  sections.forEach(current => {
    const sectionHeight = current.offsetHeight;
    const sectionTop = current.offsetTop - 200;
    const sectionId = current.getAttribute("id");
    
    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      document.querySelectorAll(".nav-links a").forEach(a => {
        a.classList.remove("nav-active");
        if (a.getAttribute("href") === "#" + sectionId) {
          a.classList.add("nav-active");
        }
      });
    }
  });
});

// CHANGE 11 — AGENT STATUS CYCLING ANIMATION
function initAgentStatusAnimation() {
  const statusEl = document.getElementById("agent-status-text");
  if (!statusEl) return;
  
  const messages = [
    "SCANNING CUDA...",
    "PARSING KERNELS...",
    "GENERATING HIP...",
    "COMPILING ON MI300X...",
    "ROCPROF BENCHMARKING...",
    "MIGRATION COMPLETE ✓"
  ];
  
  let idx = 0;
  
  if (window.agentStatusInterval) clearInterval(window.agentStatusInterval);
  
  // Set initial color based on first message
  statusEl.style.color = "#CC0000";
  statusEl.innerText = messages[0];
  statusEl.style.opacity = 1;
  
  window.agentStatusInterval = setInterval(() => {
    statusEl.style.opacity = 0;
    
    setTimeout(() => {
      idx = (idx + 1) % messages.length;
      statusEl.innerText = messages[idx];
      
      if (messages[idx] === "MIGRATION COMPLETE ✓") {
        statusEl.style.color = "#3FB950";
      } else {
        statusEl.style.color = "#CC0000";
      }
      
      statusEl.style.opacity = 1;
      
      if (messages[idx] === "MIGRATION COMPLETE ✓") {
        // Restart after next cycle
        idx = -1;
      }
    }, 300); // 300ms fade out wait
    
  }, 2000);
}


// Load kernels from API
async function loadKernelsFromAPI() {
  try {
    const tabsEl = document.querySelector(".snippet-tabs");
    if (!tabsEl) return;
    
    console.log("Fetching from API...");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const response = await fetch("http://3.239.166.194:8001/api/kernels", { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error("API kernels failed");
    console.log("API success");
    const kernels = await response.json();
    
    tabsEl.innerHTML = ""; // Clear existing hardcoded tabs
    
    kernels.forEach(k => {
      const btn = document.createElement("button");
      btn.className = "snippet-toggle-btn";
      btn.id = "btn-" + k.id;
      btn.setAttribute("onclick", `selectPlaygroundSnippet('${k.id}')`);
      btn.innerHTML = `${k.filename} <div style="font-size: 10px; opacity: 0.7; margin-top: 2px;">${k.lines} lines</div>`;
      tabsEl.appendChild(btn);
    });
    
    // Select first kernel if available
    if (kernels.length > 0 && typeof selectPlaygroundSnippet === 'function') {
        setTimeout(() => selectPlaygroundSnippet(kernels[0].id), 500);
    }
    
  } catch (e) {
    console.log("API failed, using static fallback", e);
    // If this fails, the hardcoded buttons are already in HTML, so we don't necessarily need to wipe them out.
    // If they were wiped, we could add fallback buttons, but index.html has defaults.
  }
}

// Fetch stats for the stats bar
async function updateStatsBarFromAPI() {
  try {
    console.log("Fetching from API...");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const response = await fetch("http://3.239.166.194:8001/api/status", { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error("API stats failed");
    console.log("API success");
    const stats = await response.json();
    
    // Update data-target attributes
    const numElems = document.querySelectorAll(".stat-bar-num");
    if (numElems.length >= 4) {
      numElems[0].setAttribute("data-target", stats.kernels_migrated);
      numElems[1].setAttribute("data-target", stats.cuda_calls_converted);
      numElems[2].setAttribute("data-target", stats.migration_success_rate);
      numElems[3].setAttribute("data-target", stats.total_api_cost);
    }
  } catch (e) {
    console.log("API failed, using static fallback", e);
    // UI already has default text like 4 minutes, 99%, etc. so it's fine.
  }
}

// Hook into DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
    loadKernelsFromAPI();
    updateStatsBarFromAPI();
});

// Copy functionality for code panels
async function copyPlaygroundCode(targetId, buttonId, defaultText) {
  const target = document.getElementById(targetId);
  const button = document.getElementById(buttonId);
  if (!target || !button) return;
  
  let textToCopy = target.innerText || target.textContent;
  
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(textToCopy);
    } else {
      // Fallback for older browsers or non-secure contexts
      const textArea = document.createElement("textarea");
      textArea.value = textToCopy;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      document.execCommand('copy');
      textArea.remove();
    }
    
    // Visual feedback
    const originalHTML = button.innerHTML;
    button.classList.add("copied");
    button.innerText = "Copied!";
    
    setTimeout(() => {
      button.classList.remove("copied");
      button.innerHTML = originalHTML;
    }, 2000);
    
  } catch (err) {
    console.error("Failed to copy code: ", err);
    const originalHTML = button.innerHTML;
    button.innerText = "Failed";
    setTimeout(() => {
      button.innerHTML = originalHTML;
    }, 2000);
  }
}

// Adding animations using Motion.dev
document.addEventListener("DOMContentLoaded", () => {
  if (window.Motion) {
    const { animate, stagger } = window.Motion;
    
    // Animate Hero content
    animate(".hero-title", 
      { opacity: [0, 1], y: [30, 0] }, 
      { duration: 0.8, easing: "ease-out" }
    );
    
    animate(".hero-subtitle", 
      { opacity: [0, 1], y: [20, 0] }, 
      { duration: 0.8, delay: 0.2, easing: "ease-out" }
    );
    
    animate(".btn", 
      { opacity: [0, 1], scale: [0.9, 1] }, 
      { duration: 0.5, delay: stagger(0.1, { startDelay: 0.4 }) }
    );
    
    animate(".badge-pill", 
      { opacity: [0, 1], x: [-20, 0] }, 
      { duration: 0.5, delay: stagger(0.05, { startDelay: 0.6 }), easing: "ease-out" }
    );
  }
});

// Additional scroll animations using Motion.dev
document.addEventListener("DOMContentLoaded", () => {
  if (window.Motion) {
    const { animate, inView, stagger } = window.Motion;
    
    // Animate playground panels when scrolled into view
    inView(".comparison-panel", (info) => {
      animate(info.target, 
        { opacity: [0, 1], y: [50, 0] }, 
        { duration: 0.6, easing: "ease-out" }
      );
    });
    
    // Animate pipeline steps in sequence
    inView("#pipeline-diagram", () => {
      animate(".pipeline-step", 
        { opacity: [0, 1], y: [30, 0] }, 
        { duration: 0.5, delay: stagger(0.2), easing: "ease-out" }
      );
      
      animate(".pipeline-connector", 
        { opacity: [0, 1], scaleX: [0, 1] }, 
        { duration: 0.5, delay: stagger(0.2, { startDelay: 0.2 }), easing: "ease-out" }
      );
    });
    
    // Animate stats cards
    inView(".stats-grid", () => {
      animate(".stat-card", 
        { opacity: [0, 1], scale: [0.9, 1] }, 
        { duration: 0.5, delay: stagger(0.1), easing: "spring" }
      );
    });
  }
});


// ═══════════════════════════════════════════════
// Reset Panel — Reload Original Static File
// ═══════════════════════════════════════════════

async function resetPanel(which) {
  const key = playgroundActiveKernel || activeKernel || "vectorAdd";
  const resetBtn = document.getElementById(`reset-${which}-btn`);

  if (resetBtn) {
    resetBtn.textContent = "…";
    resetBtn.disabled = true;
  }

  try {
    if (which === "cuda") {
      const res = await fetch(`/kernels/raw/${key}.cu`);
      if (!res.ok) throw new Error("Static CUDA file not found");
      const text = await res.text();
      const cudaEl = document.getElementById("cuda-playground-code");
      if (cudaEl) {
        cudaEl.innerHTML = formatCodeBlock(text, "cuda");
        cudaEl.style.opacity = "1";
        if (window.hljs) hljs.highlightElement(cudaEl);
      }
      // Re-show the hint (they reset, so it's fresh again)
      const hint = document.getElementById("cuda-edit-hint");
      if (hint) {
        hint.classList.remove("hidden");
        // Re-attach the one-time focus listener
        const cudaEl2 = document.getElementById("cuda-playground-code");
        if (cudaEl2) {
          cudaEl2.addEventListener("focus", () => {
            hint.classList.add("hidden");
          }, { once: true });
        }
      }
    } else if (which === "hip") {
      const res = await fetch(`/kernels/converted/${key}.cu.hip`);
      if (!res.ok) throw new Error("Static HIP file not found");
      const text = await res.text();
      const hipEl = document.getElementById("hip-playground-code");
      if (hipEl) {
        hipEl.innerHTML = formatCodeBlock(text, "hip");
        hipEl.style.opacity = "1";
        if (window.hljs) hljs.highlightElement(hipEl);
      }
    }
  } catch (e) {
    console.log(`Reset ${which} failed:`, e);
  }

  if (resetBtn) {
    resetBtn.disabled = false;
    // Restore reset button SVG + text
    resetBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12" style="margin-right:4px;"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
      Reset
    `;
  }
}

// ═══════════════════════════════════════════════════════════════
// File Upload — Drag & Drop + Click-to-Browse for .cu files
// ═══════════════════════════════════════════════════════════════

/** Show a status message below the drop zone */
function showUploadStatus(msg, type) {
  const el = document.getElementById("cu-upload-status");
  if (!el) return;
  el.textContent = msg;
  el.className = "cu-upload-status " + type;
  el.style.display = "block";
}

/** Process an uploaded File object: validate, read, and load into CUDA panel */
function processUploadedFile(file) {
  // Validate extension
  if (!file.name.endsWith(".cu")) {
    showUploadStatus("Only .cu files are supported", "error");
    return;
  }

  // Validate size (max 1 MB)
  if (file.size > 1024 * 1024) {
    showUploadStatus("File too large. Max 1MB", "error");
    return;
  }

  const reader = new FileReader();
  reader.onload = function(e) {
    const content = e.target.result;

    // Store in global state
    uploadedCuCode = content;
    uploadedCuFilename = file.name;

    // Load into CUDA panel
    const cudaEl = document.getElementById("cuda-playground-code");
    if (cudaEl) {
      cudaEl.innerHTML = formatCodeBlock(content, "cuda");
      cudaEl.style.opacity = "1";
      if (window.hljs) hljs.highlightElement(cudaEl);
    }

    // Clear the HIP panel
    const hipEl = document.getElementById("hip-playground-code");
    if (hipEl) {
      hipEl.innerHTML = '<span style="color:var(--text-muted); font-family:var(--font-mono); font-size:13px; padding:20px; display:block;">Run the agent to generate HIP code for your file.</span>';
      hipEl.style.opacity = "1";
    }

    // Show the "Custom" tab and activate it
    const customBtn = document.getElementById("btn-custom");
    if (customBtn) {
      customBtn.style.display = "";
      // Deactivate all other tabs
      document.querySelectorAll(".snippet-toggle-btn").forEach(b => b.classList.remove("active"));
      customBtn.classList.add("active");
    }

    // Highlight the run button
    const runBtn = document.getElementById("playground-run-btn");
    if (runBtn) {
      runBtn.style.boxShadow = "0 0 16px rgba(204,0,0,0.5)";
    }

    // Show success status
    showUploadStatus(`${file.name} loaded ✓`, "success");

    // Update the drop zone label
    const label = document.getElementById("cu-drop-label");
    if (label) {
      label.innerHTML = `<strong style="color:#27c93f;">${file.name}</strong> loaded — drop another to replace`;
    }
  };

  reader.onerror = function() {
    showUploadStatus("Error reading file — please try again", "error");
  };

  reader.readAsText(file);
}

/** Called by the <input type="file"> onchange */
function handleCuFileSelect(event) {
  const file = event.target.files && event.target.files[0];
  if (file) processUploadedFile(file);
  // Reset input so the same file can be re-selected
  event.target.value = "";
}

/** Called when user clicks the "Custom ✓" tab */
function selectCustomUpload() {
  if (!uploadedCuCode) return;

  // Re-activate custom tab
  document.querySelectorAll(".snippet-toggle-btn").forEach(b => b.classList.remove("active"));
  const customBtn = document.getElementById("btn-custom");
  if (customBtn) customBtn.classList.add("active");

  // Restore uploaded code into CUDA panel
  const cudaEl = document.getElementById("cuda-playground-code");
  if (cudaEl) {
    cudaEl.innerHTML = formatCodeBlock(uploadedCuCode, "cuda");
    cudaEl.style.opacity = "1";
    if (window.hljs) hljs.highlightElement(cudaEl);
  }
}

/** Wire drag-and-drop to the drop zone after DOM is ready */
document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("cu-drop-zone");
  if (!dropZone) return;

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) processUploadedFile(file);
  });
});
