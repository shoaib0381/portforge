
js_code = """
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
"""

with open('frontend/app.js', 'a', encoding='utf-8') as f:
    f.write(js_code)
