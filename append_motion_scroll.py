
js_code = """
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
"""

with open('frontend/app.js', 'a', encoding='utf-8') as f:
    f.write(js_code)
