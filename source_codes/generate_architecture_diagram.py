"""
MH-EWSS Architecture Diagram Generator
Generates a clean, presentation-ready architecture diagram as a PNG image.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 20))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis('off')

# ===== TITLE =====
ax.text(8, 19.3, 'MH-EWSS: System Architecture', fontsize=24, fontweight='bold',
        color='white', ha='center', va='center', fontfamily='sans-serif')
ax.text(8, 18.8, 'Mental Health Early Warning Speech System', fontsize=13,
        color='#a0a0c0', ha='center', va='center', fontfamily='sans-serif')

def draw_box(ax, x, y, w, h, color, title, subtitle='', details=None, icon=''):
    """Draw a styled rounded rectangle with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    
    # Title
    ty = y + h - 0.35 if subtitle or details else y + h/2
    ax.text(x + w/2, ty, f'{icon} {title}', fontsize=13, fontweight='bold',
            color='white', ha='center', va='center', fontfamily='sans-serif')
    
    # Subtitle
    if subtitle:
        ax.text(x + w/2, ty - 0.35, subtitle, fontsize=9.5,
                color='#e0e0e0', ha='center', va='center', fontfamily='sans-serif',
                style='italic')
    
    # Details (bullet points)
    if details:
        for i, detail in enumerate(details):
            ax.text(x + w/2, ty - 0.7 - i*0.3, f'• {detail}', fontsize=9,
                    color='#d0d0d0', ha='center', va='center', fontfamily='sans-serif')

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw a styled arrow between components"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#8899bb',
                               lw=2, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=8, color='#8899bb',
                ha='left', va='center', fontfamily='sans-serif')

def draw_db(ax, x, y, w, h, color, title, subtitle='', details=None):
    """Draw a database cylinder shape"""
    from matplotlib.patches import Ellipse
    body = FancyBboxPatch((x, y), w, h-0.25, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(body)
    ellipse_top = Ellipse((x+w/2, y+h-0.25), w, 0.5,
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.95)
    ax.add_patch(ellipse_top)
    ellipse_bot = Ellipse((x+w/2, y), w, 0.5,
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.7)
    ax.add_patch(ellipse_bot)
    
    ax.text(x+w/2, y+h/2+0.1, f'🗄️ {title}', fontsize=12, fontweight='bold',
            color='white', ha='center', va='center', fontfamily='sans-serif')
    if subtitle:
        ax.text(x+w/2, y+h/2-0.25, subtitle, fontsize=9.5,
                color='#e0e0e0', ha='center', va='center', fontfamily='sans-serif',
                style='italic')
    if details:
        for i, d in enumerate(details):
            ax.text(x+w/2, y+h/2-0.55-i*0.28, f'• {d}', fontsize=9,
                    color='#d0d0d0', ha='center', va='center', fontfamily='sans-serif')

# ============================================================
# ROW 1: AUDIO INPUT
# ============================================================
draw_box(ax, 5.5, 17.2, 5, 1.2, '#0984e3',
         'Audio Input', 'User uploads .wav file',
         icon='🎤')

# ============================================================
# ROW 2: EXTRACTION AGENTS (side by side)
# ============================================================
# Section label
ax.text(8, 15.95, '── Feature Extraction Layer ──', fontsize=10,
        color='#6c7a89', ha='center', va='center', fontfamily='sans-serif')

draw_box(ax, 1.5, 14, 5.5, 1.7, '#6c5ce7',
         'Prosody Agent', 'librosa',
         details=['13 MFCCs (Vocal Fingerprints)', '10 Acoustic Features'],
         icon='🎙️')

draw_box(ax, 9, 14, 5.5, 1.7, '#00b894',
         'Language Agent', 'SpeechRecognition + TextBlob',
         details=['Sentiment & Subjectivity', '6 Linguistic Features'],
         icon='📝')

# Arrows from Audio to Agents
draw_arrow(ax, 7, 17.2, 4.25, 15.7)
draw_arrow(ax, 9, 17.2, 11.75, 15.7)

# ============================================================
# ROW 3: TEMPORAL AGENT (DATABASE)
# ============================================================
draw_db(ax, 5, 11.2, 6, 1.8, '#2d6a4f',
        'Temporal Agent', 'SQLite Database',
        details=['Personal Baseline Tracking', 'Historical Deviation Analysis'])

# Arrows from Agents to DB
draw_arrow(ax, 4.25, 14, 7, 13.0, 'Store')
draw_arrow(ax, 11.75, 14, 9, 13.0, 'Store')

# ============================================================
# ROW 4: DUAL ENGINE AI (side by side)
# ============================================================
# Section background
section_bg = FancyBboxPatch((1, 7.3), 14, 3.2, boxstyle="round,pad=0.2",
                            facecolor='#16213e', edgecolor='#3a506b',
                            linewidth=2, alpha=0.7, linestyle='--')
ax.add_patch(section_bg)
ax.text(8, 10.15, '── Dual-Engine AI Inference ──', fontsize=10,
        color='#6c7a89', ha='center', va='center', fontfamily='sans-serif')

draw_box(ax, 1.8, 7.7, 5.2, 2.1, '#d63031',
         'Gradient Boosting', 'DAIC-WOZ + SMOTE Balanced',
         details=['23 Feature Input Vector', 'Mathematical Risk Probability', '88% Cross-Validated Accuracy'],
         icon='🤖')

draw_box(ax, 9, 7.7, 5.2, 2.1, '#e17055',
         'Gemini 2.5 Flash', 'Context-Aware LLM Reasoning',
         details=['Ingests ML Probability + Features', 'Generates Explanation & Actions', 'Outputs Structured JSON'],
         icon='🧠')

# Arrow from DB to LLM
draw_arrow(ax, 8, 11.2, 11.6, 9.8, 'Baseline')

# Arrow from DB to GBM
draw_arrow(ax, 8, 11.2, 4.4, 9.8, 'Features')

# Arrow from GBM to LLM (the key hybrid connection)
ax.annotate('', xy=(9, 8.75), xytext=(7, 8.75),
            arrowprops=dict(arrowstyle='->', color='#ffeaa7',
                           lw=2.5, connectionstyle='arc3,rad=0'))
ax.text(8, 9.05, 'Risk Score', fontsize=9, color='#ffeaa7',
        ha='center', va='center', fontweight='bold', fontfamily='sans-serif')

# ============================================================
# ROW 5: STREAMLIT DASHBOARD
# ============================================================
draw_box(ax, 3, 5, 10, 1.8, '#e84393',
         'Streamlit Dashboard', 'Real-Time Mental Health Analysis UI',
         details=['Risk Badge (Healthy / Needs Attention / Unhealthy)',
                  'ML Confidence Scores & Feature Charts'],
         icon='📊')

# Arrow from LLM to Dashboard
draw_arrow(ax, 11.6, 7.7, 10, 6.8, 'JSON Report')

# ============================================================
# ROW 6: OUTPUT
# ============================================================
draw_box(ax, 4.5, 3, 7, 1.3, '#2d3436',
         'Final Output', '',
         details=['Risk Level  •  Explanation  •  Recommendations'],
         icon='✅')

draw_arrow(ax, 8, 5, 8, 4.3)

# ============================================================
# WATERMARK
# ============================================================
ax.text(8, 2.3, 'MH-EWSS  |  DAIC-WOZ Clinical Dataset  |  LangGraph + Gemini + GBM',
        fontsize=9, color='#4a4a6a', ha='center', va='center', fontfamily='sans-serif')

# Save
plt.savefig('output/metrics/architecture_diagram.png', dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("✅ Architecture diagram saved to output/metrics/architecture_diagram.png")
