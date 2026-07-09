import os
import re

def update_js():
    with open('frontend/app.js', 'r', encoding='utf-8') as f:
        app = f.read()

    # Find the start and end of runHeroDemoAnimation
    start_idx = app.find("async function runHeroDemoAnimation()")
    end_idx = app.find("// 2. Playground Backend Runner")
    if start_idx != -1 and end_idx != -1:
        app = app[:start_idx] + "async function runHeroDemoAnimation() { /* Replaced by Change 5 and Change 11 */ }\n\n" + app[end_idx:]

    # Remove previous ENHANCEMENTS from the bottom of app.js (if any exist from before)
    # The previous script added `// Agent Status Animation` and other code
    enhancement_idx = app.find("// Agent Status Animation")
    if enhancement_idx != -1:
        app = app[:enhancement_idx]
        
    js_additions = """
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

// CHANGE 3 — ANIMATED DOT GRID BACKGROUND
function initHeroCanvas() {
  const canvas = document.getElementById("hero-bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  let width, height;
  function resize() {
    width = canvas.width = canvas.parentElement.offsetWidth;
    height = canvas.height = canvas.parentElement.offsetHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  let time = 0;
  function draw() {
    ctx.clearRect(0, 0, width, height);
    time += 0.02;
    
    // Pulse opacity using sine wave from 0.1 to 0.3
    // Base 0.1, amplitude 0.1 -> total range 0.0 to 0.2? Wait, sine goes from -1 to 1.
    // So 0.2 + (Math.sin(time) * 0.1) goes from 0.1 to 0.3. Exactly what was requested.
    const baseOpacity = 0.2;
    const pulse = Math.sin(time) * 0.1;
    ctx.fillStyle = `rgba(204, 0, 0, ${baseOpacity + pulse})`;
    
    for (let x = 0; x < width; x += 30) {
      for (let y = 0; y < height; y += 30) {
        ctx.beginPath();
        ctx.arc(x, y, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// CHANGE 4 — TYPEWRITER EFFECT ON HERO TITLE
function initTypewriter() {
  const target = document.getElementById("typewriter-target");
  if (!target) return;
  
  const text = "Minutes.";
  target.innerHTML = "";
  let i = 0;
  
  function typeChar() {
    if (i < text.length) {
      target.innerHTML = text.substring(0, i + 1);
      i++;
      setTimeout(typeChar, 80);
    } else {
      target.innerHTML = text + '<span class="typewriter-cursor"></span>';
      setTimeout(() => {
        const cursor = target.querySelector(".typewriter-cursor");
        if (cursor) cursor.remove();
      }, 3000);
    }
  }
  
  typeChar();
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
          .replace(/\\n/g, "<br>")
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
"""

    app = app + "\n" + js_additions

    with open('frontend/app.js', 'w', encoding='utf-8') as f:
        f.write(app)

update_js()
