import re

def update_app_js():
    with open('frontend/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Remove KERNEL_TEMPLATES completely.
    # It starts at the top: `// Kernels Database for Playground\nconst KERNEL_TEMPLATES = {`
    # and ends at `// Global playground state`
    
    start_str = "// Kernels Database for Playground"
    end_str = "// Global playground state"
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + content[end_idx:]
        
    # 2. Add loading spinner CSS to the top of the file (or better yet we can inject it into the DOM, but let's just append some simple logic).
    # Wait, we can inject a spinner HTML where needed.
    
    # 3. Replace selectPlaygroundSnippet(key) to use fetch
    
    old_select_func = """function selectPlaygroundSnippet(key) {
  if (isRunningAgent) return;
  activeKernel = key;
  playgroundActiveKernel = key;
  
  // Update UI active state
  document.querySelectorAll(".kernel-list-item").forEach(el => {
    el.classList.remove("active");
  });
  const activeEl = document.querySelector(`.kernel-list-item[onclick*="${key}"]`);
  if (activeEl) {
    activeEl.classList.add("active");
  }

  // Load code
  const data = KERNEL_TEMPLATES[key];
  if (data) {
    document.getElementById("playground-cuda-code").innerHTML = escapeHtml(data.cuda);
    document.getElementById("playground-hip-code").innerHTML = escapeHtml(data.hip);
    
    // Highlight
    hljs.highlightElement(document.getElementById("playground-cuda-code"));
    hljs.highlightElement(document.getElementById("playground-hip-code"));

    updateSummaryCards(data);
    updateManifestViewer(data.manifest);
    renderAstTree(data.manifest);
  }
}"""

    # We need to replace the content of selectPlaygroundSnippet
    # But wait, looking at the truncated app.js earlier, selectPlaygroundSnippet might look slightly different.
    # Let me just replace the function using regex that matches the function signature.
    
    new_select_func = """async function selectPlaygroundSnippet(key) {
  if (isRunningAgent) return;
  activeKernel = key;
  playgroundActiveKernel = key;
  
  // Update UI active state
  document.querySelectorAll(".kernel-list-item").forEach(el => {
    el.classList.remove("active");
  });
  const activeEl = document.querySelector(`.kernel-list-item[onclick*="${key}"]`);
  if (activeEl) {
    activeEl.classList.add("active");
  }

  // Show loading spinner
  const cudaEl = document.getElementById("playground-cuda-code");
  const hipEl = document.getElementById("playground-hip-code");
  const spinnerHTML = '<div class="loading-spinner" style="padding:20px; color:#999;">Loading from API... <span class="streaming-cursor"></span></div>';
  
  if (cudaEl) cudaEl.innerHTML = spinnerHTML;
  if (hipEl) hipEl.innerHTML = spinnerHTML;

  try {
    const response = await fetch(`http://3.239.166.194:8001/api/kernel/${key}`);
    if (!response.ok) throw new Error("Failed to load kernel");
    
    const data = await response.json();
    
    if (cudaEl) cudaEl.innerHTML = escapeHtml(data.cuda);
    if (hipEl) hipEl.innerHTML = escapeHtml(data.hip);
    
    // Highlight
    if (window.hljs) {
      if (cudaEl) hljs.highlightElement(cudaEl);
      if (hipEl) hljs.highlightElement(hipEl);
    }

    if (typeof updateSummaryCards === "function") updateSummaryCards(data);
    if (typeof updateManifestViewer === "function") updateManifestViewer(data.manifest);
    if (typeof renderAstTree === "function") renderAstTree(data.manifest);
    
  } catch (error) {
    console.error(error);
    if (cudaEl) cudaEl.innerHTML = '<span style="color:red;">Error loading code</span>';
    if (hipEl) hipEl.innerHTML = '<span style="color:red;">Error loading code</span>';
  }
}"""

    # Replace selectPlaygroundSnippet
    # The previous definition was likely: function selectPlaygroundSnippet(key) { ... }
    # Let's find it.
    
    start_func = content.find("function selectPlaygroundSnippet(key)")
    if start_func != -1:
        # find the end of the function. We count braces.
        brace_count = 0
        started = False
        end_func = start_func
        for i in range(start_func, len(content)):
            if content[i] == '{':
                brace_count += 1
                started = True
            elif content[i] == '}':
                brace_count -= 1
            if started and brace_count == 0:
                end_func = i + 1
                break
        
        content = content[:start_func] + new_select_func + content[end_func:]

    # 4. Modify playground-run-btn event listener to use POST /api/migrate
    
    # We want to intercept the Run Migration logic. It might be inside an event listener for #playground-run-btn
    
    # Instead of completely replacing it via AST, let's append a new event listener and override the old one,
    # or find and replace `const data = KERNEL_TEMPLATES[playgroundActiveKernel];`
    
    migrate_logic_replacement = """
    // API Migrate Logic
    const cudaCode = document.getElementById("playground-cuda-code").textContent;
    
    // Show loading
    document.getElementById("playground-hip-code").innerHTML = '<div style="padding:20px; color:#999;">Migrating via AI Agent API... <span class="streaming-cursor"></span></div>';
    
    try {
      const formData = new FormData();
      formData.append("filename", playgroundActiveKernel + ".cu");
      formData.append("cuda_code", cudaCode);
      
      const response = await fetch("http://3.239.166.194:8001/api/migrate", {
        method: "POST",
        body: formData
      });
      
      const data = await response.json();
      
      document.getElementById("playground-hip-code").innerHTML = escapeHtml(data.hip);
      if (window.hljs) {
        hljs.highlightElement(document.getElementById("playground-hip-code"));
      }
      
      if (typeof updateManifestViewer === "function" && data.manifest) {
        updateManifestViewer(data.manifest);
      }
      
      logToConsole("INFO", "Agent responded with migrated code.");
      
    } catch (e) {
      logToConsole("ERROR", "Migration failed: " + e.message);
      document.getElementById("playground-hip-code").innerHTML = '<span style="color:#CC0000;">Migration failed</span>';
    }
"""
    
    # Let's just find `const data = KERNEL_TEMPLATES[playgroundActiveKernel];` inside the run logic and replace it with fetch logic.
    # Wait, the run logic might be `function runPlaygroundMigration()` or a click listener.
    
    content = content.replace("const data = KERNEL_TEMPLATES[playgroundActiveKernel];", migrate_logic_replacement)
    
    # 5. Dynamic kernels list on sidebar. We will append a function and call it on load.
    
    dynamic_sidebar_logic = """
// Load kernels from API
async function loadKernelsFromAPI() {
  try {
    const listEl = document.getElementById("kernel-list");
    if (!listEl) return;
    
    const response = await fetch("http://3.239.166.194:8001/api/kernels");
    const kernels = await response.json();
    
    listEl.innerHTML = ""; // Clear existing hardcoded
    
    kernels.forEach(k => {
      const div = document.createElement("div");
      div.className = "kernel-list-item";
      div.setAttribute("onclick", `selectPlaygroundSnippet('${k.id}')`);
      div.innerHTML = `
        <div class="kernel-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        </div>
        <div class="kernel-info">
          <div class="kernel-name">${k.filename}</div>
          <div class="kernel-status">
            <span class="status-indicator status-done"></span> Migrated
          </div>
        </div>
      `;
      listEl.appendChild(div);
    });
    
    // Select first kernel if available
    if (kernels.length > 0 && typeof selectPlaygroundSnippet === 'function') {
        setTimeout(() => selectPlaygroundSnippet(kernels[0].id), 500);
    }
    
  } catch (e) {
    console.error("Failed to load kernels list", e);
  }
}

// Fetch stats for the stats bar
async function updateStatsBarFromAPI() {
  try {
    const response = await fetch("http://3.239.166.194:8001/api/status");
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
    console.error("Failed to load stats", e);
  }
}

// Hook into DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
    loadKernelsFromAPI();
    updateStatsBarFromAPI();
});
"""
    content += "\n" + dynamic_sidebar_logic

    with open('frontend/app.js', 'w', encoding='utf-8') as f:
        f.write(content)

update_app_js()
