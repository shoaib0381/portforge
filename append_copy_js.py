
# Copy functionality for code panels
js_code = """
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
"""

with open('frontend/app.js', 'a', encoding='utf-8') as f:
    f.write(js_code)
