import os
import re

def update_html():
    with open('frontend/index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # 1. Remove Verifier Section
    html = re.sub(r'<!-- Footer Verifier Section -->.*?<section id="verifier".*?</section>', '', html, flags=re.DOTALL)
    html = re.sub(r'<section id="verifier".*?</section>', '', html, flags=re.DOTALL) # Fallback

    # 2. Logo Icon
    svg_logo = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 32 24" fill="none" class="logo-symbol" style="background:transparent;"><path d="M10 6L4 12L10 18" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 20L18 4" stroke="white" stroke-width="2" stroke-linecap="round"/><path d="M18 12H30M30 12L24 6M30 12L24 18" stroke="#ED1C24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    html = re.sub(r'<div class="logo-symbol">.*?</div>', svg_logo, html, flags=re.DOTALL)
    html = html.replace('<div class="logo-symbol"></div>', svg_logo)

    # 3. Typing Animation CSS class
    html = html.replace('<span class="amd-gradient">Minutes.</span>', '<span class="amd-gradient type-animate">Minutes.</span>')
    html = html.replace('<span class="amd-gradient">Minutes.</span>', '<span class="amd-gradient type-animate">Minutes.</span>') # Fallback if missed

    # 4. Fade in up class
    html = html.replace('class="team-card"', 'class="team-card fade-in-up"')
    html = html.replace('class="pipeline-step"', 'class="pipeline-step fade-in-up"')
    html = html.replace('class="stat-card"', 'class="stat-card fade-in-up"')

    # 6. Stats Bar
    stats_html = '''
  <!-- Stats Bar -->
  <section class="stats-bar-section fade-in-up" id="stats-bar-section">
    <div class="container stats-bar">
      <div class="stat-bar-item">
        <div class="stat-bar-num" data-val="5">0</div>
        <div class="stat-bar-label">Kernels Migrated</div>
      </div>
      <div class="stat-bar-item">
        <div class="stat-bar-num" data-val="152">0</div>
        <div class="stat-bar-label">CUDA Calls Converted</div>
      </div>
      <div class="stat-bar-item">
        <div class="stat-bar-num" data-val="100" data-suffix="%">0%</div>
        <div class="stat-bar-label">Compilation Success</div>
      </div>
      <div class="stat-bar-item">
        <div class="stat-bar-num" data-val="0.003" data-prefix="$" data-decimals="3">0</div>
        <div class="stat-bar-label">Total API Cost</div>
      </div>
    </div>
  </section>
'''
    # Insert before features section
    html = html.replace('<section id="features"', stats_html + '\n  <section id="features"')

    with open('frontend/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

def update_css():
    with open('frontend/style.css', 'r', encoding='utf-8') as f:
        css = f.read()
        
    css += '''
/* ENHANCEMENTS */

/* 3. Typing Animation */
.amd-gradient.type-animate {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  border-right: 0.15em solid #ED1C24;
  animation: 
    typing 1.5s steps(8, end) forwards,
    blink-caret 0.75s step-end infinite;
  max-width: 0;
  vertical-align: bottom;
}
@keyframes typing {
  from { max-width: 0 }
  to { max-width: 100% }
}
@keyframes blink-caret {
  from, to { border-color: transparent }
  50% { border-color: #ED1C24; }
}

/* 4. Scroll Animations */
.fade-in-up {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}
.fade-in-up.visible {
  opacity: 1;
  transform: translateY(0);
}

/* 5. Agent Status Transition */
#agent-status-text {
  transition: opacity 0.4s ease;
}

/* 6. Stats Bar */
.stats-bar-section {
  background-color: #0a0a0f;
  border-top: 1px solid #222;
  border-bottom: 1px solid #222;
  padding: 32px 0;
  margin: 40px 0;
}
.stats-bar {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  align-items: center;
  flex-wrap: wrap;
  gap: 24px;
}
.stat-bar-item {
  text-align: center;
}
.stat-bar-num {
  font-size: 2.5rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 8px;
}
.stat-bar-label {
  font-size: 0.9rem;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* 7. Smooth Scroll */
html {
  scroll-behavior: smooth;
}

/* 8. Active Nav */
.nav-links a {
  transition: color 0.3s ease;
}
.nav-links a.active {
  color: #ED1C24 !important;
}
'''
    with open('frontend/style.css', 'w', encoding='utf-8') as f:
        f.write(css)

def update_js():
    with open('frontend/app.js', 'r', encoding='utf-8') as f:
        app = f.read()

    # 4. Scroll Animations Observer & 6. Stats Observer
    # We will append logic at the end or replace existing empty setupStatsObserver/setupSectionObserver
    
    # 5. Agent status animation
    agent_status_js = '''
// Agent Status Animation
function initAgentStatusAnimation() {
  const statusEl = document.getElementById("agent-status-text");
  if(!statusEl) return;
  const messages = [
    "SCANNING CUDA...",
    "ANALYZING KERNELS...",
    "GENERATING HIP...",
    "COMPILING ON MI300X...",
    "BENCHMARK COMPLETE"
  ];
  let idx = 0;
  setInterval(() => {
    statusEl.style.opacity = 0;
    setTimeout(() => {
      idx = (idx + 1) % messages.length;
      statusEl.innerText = messages[idx];
      statusEl.style.opacity = 1;
    }, 400);
  }, 2000);
}
initAgentStatusAnimation();
'''
    # We'll just append it to the file so it runs
    app += agent_status_js

    # Add Intersection Observer for .fade-in-up
    fade_in_js = '''
// Fade In Up Observer
const fadeElems = document.querySelectorAll('.fade-in-up');
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if(entry.isIntersecting) {
      entry.target.classList.add('visible');
      // If it's the stats bar, trigger counter
      if (entry.target.classList.contains('stats-bar-section')) {
         triggerStatsCount();
         fadeObserver.unobserve(entry.target);
      }
    }
  });
}, { threshold: 0.1 });
fadeElems.forEach(el => fadeObserver.observe(el));

function triggerStatsCount() {
  const statNums = document.querySelectorAll('.stat-bar-num');
  statNums.forEach(el => {
    const target = parseFloat(el.getAttribute('data-val'));
    const suffix = el.getAttribute('data-suffix') || '';
    const prefix = el.getAttribute('data-prefix') || '';
    const decimals = parseInt(el.getAttribute('data-decimals')) || 0;
    
    let current = 0;
    const increment = target / 40;
    const timer = setInterval(() => {
      current += increment;
      if(current >= target) {
        current = target;
        clearInterval(timer);
      }
      el.innerText = prefix + current.toFixed(decimals) + suffix;
    }, 30);
  });
}
'''
    app += fade_in_js

    # Active Nav Highlight
    # I'll replace the existing scrollSpy if it exists, or append it.
    if 'function scrollSpy(' in app:
        # replace the function body
        pass
    nav_js = '''
function updateActiveNav() {
  const sections = document.querySelectorAll("section[id]");
  const scrollY = window.scrollY;
  
  sections.forEach(current => {
    const sectionHeight = current.offsetHeight;
    const sectionTop = current.offsetTop - 150;
    const sectionId = current.getAttribute("id");
    
    if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
      document.querySelectorAll(".nav-links a").forEach(a => {
        a.classList.remove("active");
        if(a.getAttribute("href") === "#" + sectionId) {
          a.classList.add("active");
        }
      });
    }
  });
}
window.addEventListener('scroll', updateActiveNav);
'''
    app += nav_js

    with open('frontend/app.js', 'w', encoding='utf-8') as f:
        f.write(app)

update_html()
update_css()
update_js()
