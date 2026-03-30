"""
Civic AI Analyzer — premium Streamlit UI for complaint classification + rule-based priority.
Run: streamlit run app.py
"""


from __future__ import annotations

import csv
import html
import random
import re
import time
import traceback
from pathlib import Path

import streamlit as st

from civic_nlp import analyze_complaint, ensure_nltk, load_pipeline

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Civic AI Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium theme CSS (Stripe / Notion / Linear inspired)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

html, body, [class*="css"]  {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

#root > div:first-child { background: transparent !important; }

.stApp {
  background: radial-gradient(ellipse 120% 80% at 50% -20%, #1e3a5f 0%, #0a0e17 45%, #020308 100%) !important;
  min-height: 100vh;
  font-size: 16px;
  line-height: 1.55;
  color: #e2e8f0;
}

/* Streamlit native helpers */
div[data-testid="stCaption"] {
  color: #94a3b8 !important;
  font-size: 0.9rem !important;
  line-height: 1.6 !important;
  letter-spacing: 0.01em;
  max-width: 58ch;
}
p[data-testid="stMarkdownContainer"] {
  line-height: 1.6;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stToolbar"] { display: none; }

div[data-testid="stAlert"] {
  font-size: 0.95rem !important;
  line-height: 1.55 !important;
}

.block-container {
  max-width: 960px !important;
  padding-top: 2rem !important;
  padding-bottom: 4rem !important;
}

/* Primary button */
div.stButton > button:first-child {
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%) !important;
  color: #f8fafc !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.65rem 1.75rem !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  letter-spacing: 0.02em;
  box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35), inset 0 1px 0 rgba(255,255,255,0.15);
  transition: transform 0.2s ease, box-shadow 0.25s ease, filter 0.2s ease !important;
}
div.stButton > button:first-child:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 8px 32px rgba(99, 102, 241, 0.45);
  filter: brightness(1.08);
}
div.stButton > button:first-child:active {
  transform: scale(0.98);
}

/* Secondary / ghost buttons */
button[kind="secondary"] {
  background: rgba(255,255,255,0.06) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(148, 163, 184, 0.2) !important;
  border-radius: 12px !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
  transition: background 0.2s, border-color 0.2s, transform 0.15s !important;
}
button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.1) !important;
  border-color: rgba(148, 163, 184, 0.35) !important;
}

/* Text area */
.stTextArea textarea {
  background: rgba(15, 23, 42, 0.65) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  border-radius: 16px !important;
  color: #f8fafc !important;
  font-size: 1.08rem !important;
  line-height: 1.65 !important;
  letter-spacing: 0.015em;
  padding: 1.15rem 1.3rem !important;
  box-shadow: inset 0 2px 12px rgba(0,0,0,0.25);
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: rgba(99, 102, 241, 0.55) !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15), inset 0 2px 12px rgba(0,0,0,0.25);
}

.stTextArea label { color: #94a3b8 !important; font-weight: 500 !important; }

/* Hero */
.hero-title {
  font-size: clamp(2rem, 5vw, 2.75rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 40%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.35rem;
  text-align: center;
}
.hero-sub {
  color: #94a3b8;
  font-size: 1.1rem;
  text-align: center;
  margin-bottom: 2rem;
  font-weight: 500;
  line-height: 1.55;
  max-width: 36rem;
  margin-left: auto;
  margin-right: auto;
}

/* Glass cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-top: 1.5rem;
}
@media (max-width: 768px) {
  .card-grid { grid-template-columns: 1fr; }
}

.glass-card {
  position: relative;
  border-radius: 20px;
  padding: 1.35rem 1.5rem;
  background: linear-gradient(145deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  overflow: hidden;
}
.glass-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 16px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
  border-color: rgba(255,255,255,0.12);
}

.card-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #94a3b8;
  font-weight: 600;
  margin-bottom: 0.55rem;
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.card-value {
  font-size: 1.7rem;
  font-weight: 700;
  color: #f8fafc;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

/* Muted lines inside result cards */
.card-muted {
  color: #94a3b8;
  font-size: 0.9rem;
  margin-top: 0.4rem;
  line-height: 1.5;
}
.card-muted-sm {
  color: #94a3b8;
  font-size: 0.84rem;
  margin-top: 0.35rem;
  line-height: 1.5;
}
.pct-side-label {
  color: #cbd5e1;
  font-size: 0.95rem;
  font-weight: 500;
}
.pri-denominator {
  font-size: 1.05rem;
  color: #94a3b8;
  font-weight: 500;
}

/* Category accents */
.cat-road { --accent: #f97316; --accent-glow: rgba(249, 115, 22, 0.35); }
.cat-water { --accent: #38bdf8; --accent-glow: rgba(56, 189, 248, 0.35); }
.cat-electricity { --accent: #facc15; --accent-glow: rgba(250, 204, 21, 0.35); }
.cat-garbage { --accent: #4ade80; --accent-glow: rgba(74, 222, 128, 0.35); }
.cat-default { --accent: #a78bfa; --accent-glow: rgba(167, 139, 250, 0.3); }

.glass-card.cat-strip::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  border-radius: 20px 0 0 20px;
  background: var(--accent);
  box-shadow: 0 0 20px var(--accent-glow);
}

/* Progress bar */
.progress-wrap {
  margin-top: 0.75rem;
  height: 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.35);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.06);
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #6366f1, #22d3ee);
  box-shadow: 0 0 16px rgba(99, 102, 241, 0.5);
  transition: width 1.1s cubic-bezier(0.22, 1, 0.36, 1);
}

/* Priority badges */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.35rem 0.85rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.badge-high {
  background: rgba(239, 68, 68, 0.2);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.45);
  box-shadow: 0 0 24px rgba(239, 68, 68, 0.25);
}
.badge-medium {
  background: rgba(245, 158, 11, 0.18);
  color: #fcd34d;
  border: 1px solid rgba(245, 158, 11, 0.4);
  box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
}
.badge-low {
  background: rgba(34, 197, 94, 0.15);
  color: #86efac;
  border: 1px solid rgba(34, 197, 94, 0.35);
  box-shadow: 0 0 16px rgba(34, 197, 94, 0.15);
}

.pri-score {
  margin-top: 0.65rem;
  font-size: 2rem;
  font-weight: 700;
  color: #f1f5f9;
  line-height: 1.15;
}

.explain-body {
  color: #cbd5e1;
  font-size: 0.98rem;
  line-height: 1.65;
  margin-top: 0.35rem;
}
.explain-body mark {
  font-weight: 600;
  font-size: 0.92em;
  padding: 0.1rem 0.4rem;
  border-radius: 6px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}
.explain-body mark.hl {
  background: rgba(129, 140, 248, 0.2);
  color: #e0e7ff;
  border: 1px solid rgba(165, 180, 252, 0.25);
}
.explain-body mark.hl-danger {
  background: rgba(239, 68, 68, 0.15);
  color: #fecaca;
  border: 1px solid rgba(248, 113, 113, 0.3);
}
.explain-body mark.hl-warn {
  background: rgba(245, 158, 11, 0.12);
  color: #fde68a;
  border: 1px solid rgba(251, 191, 36, 0.28);
}
.explain-body mark.hl-ok {
  background: rgba(34, 197, 94, 0.12);
  color: #bbf7d0;
  border: 1px solid rgba(74, 222, 128, 0.28);
}

/* Explanation card — premium narrative */
.glass-card.explain-glass {
  padding: 1.25rem 1.35rem 1.35rem;
}
.glass-card.explain-glass .card-label {
  color: #a5b4fc;
  margin-bottom: 0.35rem;
}
.explain-stack {
  margin-top: 0.2rem;
  max-height: 19rem;
  overflow-y: auto;
  padding-right: 0.4rem;
}
.explain-stack::-webkit-scrollbar {
  width: 5px;
}
.explain-stack::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(129, 140, 248, 0.45), rgba(99, 102, 241, 0.25));
  border-radius: 99px;
}
.explain-head-block {
  margin-bottom: 1rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.06) 0%, transparent 100%);
  margin-left: -0.15rem;
  margin-right: -0.15rem;
  padding-left: 0.15rem;
  padding-right: 0.15rem;
  border-radius: 12px 12px 0 0;
}
.explain-headline {
  font-size: 1.14rem;
  font-weight: 600;
  color: #f8fafc;
  letter-spacing: -0.025em;
  line-height: 1.4;
  margin: 0;
}
.explain-ul {
  list-style: none;
  padding: 0;
  margin: 0 0 0.85rem 0;
}
.explain-li {
  list-style: none;
  margin: 0 0 0.65rem 0;
  padding: 0;
  background: none;
  border: none;
}
.explain-li-inner {
  border-radius: 14px;
  overflow: hidden;
  background: linear-gradient(155deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.92) 100%);
  border: 1px solid rgba(148, 163, 184, 0.11);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.04);
  transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.2s ease;
}
.explain-li:hover .explain-li-inner {
  border-color: rgba(148, 163, 184, 0.18);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.explain-li--high .explain-li-inner {
  box-shadow: inset 3px 0 0 0 #f87171, 0 4px 20px rgba(0, 0, 0, 0.22);
}
.explain-li--time .explain-li-inner {
  box-shadow: inset 3px 0 0 0 #38bdf8, 0 4px 20px rgba(0, 0, 0, 0.22);
}
.explain-li--med .explain-li-inner {
  box-shadow: inset 3px 0 0 0 #fbbf24, 0 4px 20px rgba(0, 0, 0, 0.22);
}
.explain-li--neutral .explain-li-inner {
  box-shadow: inset 3px 0 0 0 #818cf8, 0 4px 20px rgba(0, 0, 0, 0.22);
}
.explain-li-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.85rem 0.45rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.08);
  background: rgba(2, 6, 23, 0.35);
}
.explain-pill {
  display: inline-flex;
  align-items: center;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.28rem 0.55rem;
  border-radius: 6px;
  border: 1px solid transparent;
}
.explain-pill-high {
  color: #fecaca;
  background: rgba(248, 113, 113, 0.12);
  border-color: rgba(248, 113, 113, 0.28);
}
.explain-pill-time {
  color: #7dd3fc;
  background: rgba(56, 189, 248, 0.1);
  border-color: rgba(56, 189, 248, 0.25);
}
.explain-pill-med {
  color: #fde68a;
  background: rgba(251, 191, 36, 0.1);
  border-color: rgba(251, 191, 36, 0.28);
}
.explain-pill-neutral {
  color: #c7d2fe;
  background: rgba(129, 140, 248, 0.12);
  border-color: rgba(129, 140, 248, 0.25);
}
.explain-li-body {
  padding: 0.75rem 0.95rem 0.85rem;
  color: #e2e8f0;
  font-size: 0.95rem;
  line-height: 1.68;
  letter-spacing: 0.012em;
}
/* Keyword chips — soft glass, not harsh yellow */
.explain-li-body mark,
.explain-footer mark {
  font-weight: 600;
  font-size: 0.9em;
  padding: 0.1rem 0.4rem;
  border-radius: 6px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}
.explain-li-body mark.hl,
.explain-footer mark.hl {
  background: rgba(129, 140, 248, 0.2);
  color: #e0e7ff;
  border: 1px solid rgba(165, 180, 252, 0.25);
}
.explain-li-body mark.hl-danger,
.explain-footer mark.hl-danger {
  background: rgba(239, 68, 68, 0.15);
  color: #fecaca;
  border: 1px solid rgba(248, 113, 113, 0.3);
}
.explain-li-body mark.hl-warn,
.explain-footer mark.hl-warn {
  background: rgba(245, 158, 11, 0.12);
  color: #fde68a;
  border: 1px solid rgba(251, 191, 36, 0.28);
}
.explain-li-body mark.hl-ok,
.explain-footer mark.hl-ok {
  background: rgba(34, 197, 94, 0.12);
  color: #bbf7d0;
  border: 1px solid rgba(74, 222, 128, 0.28);
}
.explain-footer {
  font-size: 0.87rem;
  color: #94a3b8;
  line-height: 1.65;
  padding: 0.8rem 1rem;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.07) 0%, rgba(30, 41, 59, 0.4) 100%);
  border: 1px solid rgba(129, 140, 248, 0.15);
  margin-top: 0.25rem;
}

/* Results animation */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
.results-animate {
  animation: fadeUp 0.65s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

.sample-row { margin-top: 1rem; }
.sample-title {
  color: #94a3b8;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.11em;
  font-weight: 600;
  margin: 0 !important;
}
.sample-stack {
  margin-bottom: 0.15rem;
}
.sample-card {
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  background: linear-gradient(165deg, rgba(30, 41, 59, 0.55) 0%, rgba(15, 23, 42, 0.92) 100%);
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.04);
  min-height: 7.5rem;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.sample-card:hover {
  border-color: rgba(148, 163, 184, 0.22);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
}
.sample-card-road { border-top: 3px solid #f97316; }
.sample-card-water { border-top: 3px solid #38bdf8; }
.sample-card-electricity { border-top: 3px solid #facc15; }
.sample-card-garbage { border-top: 3px solid #4ade80; }
.sample-card-default { border-top: 3px solid #a78bfa; }
.sample-cat-label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 0.5rem;
}
.sample-cat-emoji {
  font-size: 1.05rem;
  line-height: 1;
}
.sample-cat-name {
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #f1f5f9;
}
.sample-snippet {
  font-size: 0.875rem;
  line-height: 1.62;
  color: #cbd5e1;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  letter-spacing: 0.01em;
}
.sample-stack + div .stButton > button {
  margin-top: 0.45rem !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.55rem !important;
  min-height: 2.25rem !important;
  border-radius: 10px !important;
}

.footer-note {
  text-align: center;
  color: #64748b;
  font-size: 0.88rem;
  line-height: 1.55;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(148, 163, 184, 0.12);
}

.pct-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-top: 0.25rem;
}
.pct-num {
  font-size: 1.35rem;
  font-weight: 700;
  background: linear-gradient(90deg, #a5b4fc, #67e8f9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# NLP bootstrap — cache only joblib load; always call ensure_nltk() each run
# (after a hot-reload, module globals like STOPWORDS reset but the cache still hits)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_pipeline_cached():
    return load_pipeline()


def _get_pipeline():
    ensure_nltk()
    return _load_pipeline_cached()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CAT_CLASS = {
    "road": "cat-road",
    "water": "cat-water",
    "electricity": "cat-electricity",
    "garbage": "cat-garbage",
}

CAT_ICON = {
    "road": "🛣️",
    "water": "💧",
    "electricity": "⚡",
    "garbage": "♻️",
}

HIGHLIGHT_TERMS = [
    "exposed wire",
    "frequent issue",
    "more than a week",
    "over a week",
    "past week",
    "overflowing",
    "overflow",
    "leakage",
    "leaking",
    "frequently",
    "frequent",
    "accidents",
    "accident",
    "fires",
    "fire",
    "risks",
    "risk",
    "danger",
    "since",
    "timeline",
    "days",
    "day",
    "HIGH",
    "MEDIUM",
    "LOW",
    "general complaint",
    "long duration",
    "phrase",
]


def highlight_explanation(text: str) -> str:
    esc = html.escape(text)
    terms = sorted(HIGHLIGHT_TERMS, key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in terms)
    pat = re.compile(f"({alt})\\b", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        matched = m.group(1)
        cls = "hl"
        low = matched.lower()
        if any(x in low for x in ("high", "danger", "fire", "accident", "risk", "exposed")):
            cls = "hl-danger"
        elif any(x in low for x in ("medium", "leak", "overflow", "frequent")):
            cls = "hl-warn"
        elif "low" in low or "general" in low:
            cls = "hl-ok"
        return f'<mark class="{cls}">{html.escape(matched)}</mark>'

    return pat.sub(repl, esc)


def _explain_li_variant(bullet: str, index: int) -> tuple[str, str, str]:
    """Return (li_modifier_class, pill_class, pill_label) for layout + tier chip."""
    t = bullet.lower().strip()
    if t.startswith("no high-tier") or "no high-tier" in t[:40]:
        return ("explain-li--high", "explain-pill-high", "High tier")
    if t.startswith("no long timeline") or ("long timeline:" in t and "we look" in t):
        return ("explain-li--time", "explain-pill-time", "Duration rules")
    if t.startswith("no medium-tier") or "no medium-tier" in t[:40]:
        return ("explain-li--med", "explain-pill-med", "Medium tier")
    if t.startswith("safety") or ("safety" in t and "risk" in t and "language" in t):
        return ("explain-li--high", "explain-pill-high", "Safety")
    if "electrical hazard" in t:
        return ("explain-li--high", "explain-pill-high", "Electrical")
    if t.startswith("long timeline:"):
        return ("explain-li--time", "explain-pill-time", "Timeline")
    if "service impact" in t:
        return ("explain-li--med", "explain-pill-med", "Service impact")
    if t.startswith("recurrence"):
        return ("explain-li--med", "explain-pill-med", "Recurrence")
    if "priority is rule-based" in t or "adjust keyword" in t:
        return ("explain-li--neutral", "explain-pill-neutral", "Context")
    return ("explain-li--neutral", "explain-pill-neutral", f"Check {index + 1}")


def explanation_block_html(pri: dict) -> str:
    """Rich explanation card: headline, tier-labeled bullets, footer; fallback to legacy signals."""
    exp = pri.get("explanation")
    if not exp:
        joined = "; ".join(pri.get("signals") or [])
        return f'<div class="explain-body">{highlight_explanation(joined)}</div>'

    headline = html.escape(str(exp.get("headline") or ""))
    parts = [
        '<div class="explain-stack">',
        '<div class="explain-head-block">',
        f'<p class="explain-headline">{headline}</p>',
        "</div>",
        '<ul class="explain-ul">',
    ]
    for i, b in enumerate(exp.get("bullets") or []):
        li_mod, pill_cls, pill_lbl = _explain_li_variant(str(b), i)
        inner = highlight_explanation(str(b))
        pill_txt = html.escape(pill_lbl)
        parts.append(
            f'<li class="explain-li {li_mod}">'
            '<div class="explain-li-inner">'
            '<div class="explain-li-toolbar">'
            f'<span class="explain-pill {pill_cls}">{pill_txt}</span>'
            "</div>"
            f'<div class="explain-li-body">{inner}</div>'
            "</div></li>"
        )
    parts.append("</ul>")
    foot = exp.get("footer")
    if foot:
        parts.append(f'<div class="explain-footer">{highlight_explanation(str(foot))}</div>')
    parts.append("</div>")
    return "".join(parts)


# Embedded backup if complaints.csv is missing or empty
_FALLBACK_POOL: list[tuple[str, str]] = [
    ("The main road near the bus stop is completely broken and full of deep potholes for over a week.", "road"),
    ("There has been no water supply in our area since 5 days and the tanker never comes.", "water"),
    ("Electrical wires are hanging loose near our market and pose a danger to pedestrians.", "electricity"),
    ("Garbage has not been collected from the street and the bins are overflowing with a bad smell.", "garbage"),
    ("A live electric wire has fallen on the wet road near the market and needs urgent attention.", "electricity"),
    ("Sewage water is mixing with drinking water supply near our housing society.", "water"),
    ("The footpath along the main road has been dug up and left unrepaired for weeks.", "road"),
    ("Piles of uncollected waste outside our apartment building are breeding mosquitoes.", "garbage"),
    ("Power cut is going on continuously for the past 6 hours in our locality.", "electricity"),
    ("We have not received piped water supply for the past 3 days in our block.", "water"),
    ("Huge crater has formed on the service road causing traffic jams daily.", "road"),
    ("Trash bins near the park are overflowing and no one has come to clear them.", "garbage"),
]


def _load_complaint_pool() -> list[tuple[str, str]]:
    """All (text, category) rows for random sample buttons."""
    path = Path(__file__).resolve().parent / "complaints.csv"
    rows: list[tuple[str, str]] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (row.get("text") or "").strip()
                c = (row.get("category") or "").strip().lower()
                if t and c:
                    rows.append((t, c))
    return rows if rows else list(_FALLBACK_POOL)


def _sample_preview(text: str, max_chars: int = 148) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    cut = t[: max_chars + 1]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(".,;:") + "…"


def _sample_card_html(category: str, text: str) -> str:
    cat = (category or "").strip().lower()
    skin = cat if cat in CAT_CLASS else "default"
    icon = html.escape(CAT_ICON.get(cat, "📌"))
    title = html.escape(cat.capitalize())
    body = html.escape(_sample_preview(text))
    return (
        f'<div class="sample-card sample-card-{skin}">'
        f'<div class="sample-cat-label">'
        f'<span class="sample-cat-emoji">{icon}</span>'
        f'<span class="sample-cat-name">{title}</span>'
        f"</div>"
        f'<p class="sample-snippet">{body}</p>'
        f"</div>"
    )


def _draw_sample_batch(pool: list[tuple[str, str]], k: int = 4) -> list[tuple[str, str]]:
    k = min(k, len(pool))
    return random.sample(pool, k=k) if k else []


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "complaint_input" not in st.session_state:
    st.session_state.complaint_input = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "result_tick" not in st.session_state:
    st.session_state.result_tick = 0
if "_clear_complaint" not in st.session_state:
    st.session_state._clear_complaint = False
if "_complaint_override" not in st.session_state:
    st.session_state._complaint_override = None

_pool = _load_complaint_pool()
if "sample_batch" not in st.session_state:
    st.session_state.sample_batch = _draw_sample_batch(_pool, k=4)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">Civic AI Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Smart complaint classification &amp; prioritization</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model error (friendly)
# ---------------------------------------------------------------------------
try:
    pipeline = _get_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Input — apply programmatic changes BEFORE the text_area (widget owns key)
# ---------------------------------------------------------------------------
if st.session_state._clear_complaint:
    st.session_state.complaint_input = ""
    st.session_state.result = None
    st.session_state._clear_complaint = False
elif st.session_state._complaint_override is not None:
    st.session_state.complaint_input = st.session_state._complaint_override
    st.session_state._complaint_override = None
    st.session_state.result = None

st.text_area(
    "Your complaint",
    height=200,
    placeholder="Describe your issue…",
    key="complaint_input",
    label_visibility="collapsed",
)

c1, c2, _ = st.columns([1.1, 1.1, 2])
with c1:
    analyze = st.button("✨ Analyze", type="primary", use_container_width=True)
with c2:
    if st.button("Clear", type="secondary", use_container_width=True):
        st.session_state._clear_complaint = True
        st.rerun()

hdr_left, hdr_right = st.columns([0.82, 0.18])
with hdr_left:
    st.markdown('<p class="sample-title">Sample inputs</p>', unsafe_allow_html=True)
    st.caption(
        'Each card shows the dataset category. "Load example" fills the text box; "New set" draws four new random complaints.'
    )
with hdr_right:
    if st.button(
        "🎲 New set",
        help="Draw 4 new random complaints from the pool",
        key="sample_shuffle",
        use_container_width=True,
    ):
        st.session_state.sample_batch = _draw_sample_batch(_pool, k=4)
        st.rerun()

s_row = st.columns(4)
for i in range(4):
    with s_row[i]:
        if i < len(st.session_state.sample_batch):
            text, cat = st.session_state.sample_batch[i]
            st.markdown(
                f'<div class="sample-stack">{_sample_card_html(cat, text)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Load example", key=f"sample_{i}", use_container_width=True):
                st.session_state._complaint_override = text
                st.rerun()

# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
if analyze:
    raw = (st.session_state.get("complaint_input") or "").strip()
    if not raw:
        st.warning("Please describe your issue before analyzing.")
    else:
        with st.spinner("Analyzing with NLP + ML + rules…"):
            time.sleep(0.35)
            try:
                st.session_state.result = analyze_complaint(raw, pipeline)
                st.session_state.result_tick = st.session_state.get("result_tick", 0) + 1
            except Exception as ex:
                detail = str(ex).strip() or repr(ex)
                st.error(f"Analysis failed: {detail}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
                st.session_state.result = None

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.result
if res:
    cat = res["category"]
    conf_pct = round(res["confidence"] * 100, 1)
    pri = res["priority"]
    lvl = pri["level"].lower()
    score = pri["score_1_10"]
    explain_html = explanation_block_html(pri)

    cat_cls = CAT_CLASS.get(cat, "cat-default")
    icon = CAT_ICON.get(cat, "📌")

    badge_cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(
        lvl, "badge-low"
    )
    badge_label = lvl.upper()

    tick = st.session_state.result_tick
    anim_class = "results-animate"

    st.markdown(
        f"""
<div class="{anim_class}" data-result-tick="{tick}">
  <div class="card-grid">
    <div class="glass-card cat-strip {cat_cls}">
      <div class="card-label"><span>{icon}</span> Category</div>
      <div class="card-value" style="text-transform: capitalize;">{html.escape(cat)}</div>
      <div class="card-muted">Multinomial NB + TF-IDF</div>
    </div>
    <div class="glass-card">
      <div class="card-label">📊 Confidence</div>
      <div class="pct-label">
        <span class="pct-side-label">Model certainty</span>
        <span class="pct-num">{conf_pct}%</span>
      </div>
      <div class="progress-wrap">
        <div class="progress-fill" style="width: {conf_pct}%;"></div>
      </div>
    </div>
    <div class="glass-card">
      <div class="card-label">🚨 Priority</div>
      <div><span class="badge {badge_cls}">{badge_label}</span></div>
      <div class="pri-score">{score}<span class="pri-denominator"> / 10</span></div>
      <div class="card-muted-sm">Rule-based signals on your text</div>
    </div>
    <div class="glass-card explain-glass">
      <div class="card-label">🧠 Explanation</div>
      {explain_html}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="footer-note">Built with NLP + ML + Rule-based Intelligence</p>',
    unsafe_allow_html=True,
)
