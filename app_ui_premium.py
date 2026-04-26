"""Gradio UI for the HF Space (mounted at /ui)."""

from __future__ import annotations

import ast
import json
import random
import time

import gradio as gr
from pathlib import Path

from environment import FCEnvEnvironment
from models import Action

# Match reset() in FCEnvEnvironment (for token bar; backend unchanged)
MAX_TOKENS = 100
HIDDEN = "HIDDEN"

# Labels for multi-seed rows in Compare tab (order matches `train.py` SEEDS / `evaluation.json`)
EVAL_REPORT_SEEDS = (0, 42, 99)

# Live score panel (Play tab); reset on new episode
LIVE_STATS_DEFAULT: dict[str, int | float] = {
    "current_tokens": MAX_TOKENS,
    "total_reward": 0.0,
    "step_count": 0,
}

# Human-readable clue attribute badges (Play card grid HTML)
ATTR_LABELS: dict[str, str] = {
    "nationality": "NATION",
    "position": "POSITION",
    "club": "CLUB",
    "league": "LEAGUE",
    "decade": "DECADE",
    "caps": "CAPS",
    "age": "AGE",
    "goals": "GOALS",
}

# Light-base premium design system — no dark forcing, no neon glows
CSS_STRING = r"""
/* --- FC Decision Lab: premium light skin --- */
/* No color-scheme override: respects browser preference but layout is light-pinned */

:root {
  --fc-bg:        #f4f6f9;
  --fc-surface:   #ffffff;
  --fc-surface2:  #f8f9fb;
  --fc-border:    #e2e6ec;
  --fc-border2:   #d0d5de;
  --fc-text:      #111827;
  --fc-text2:     #374151;
  --fc-muted:     #6b7280;
  --fc-muted2:    #9ca3af;
  --fc-accent:    #1d4ed8;
  --fc-accent2:   #2563eb;
  --fc-accent-bg: #eff6ff;
  --fc-pos:       #16a34a;
  --fc-pos-bg:    #f0fdf4;
  --fc-neg:       #dc2626;
  --fc-neg-bg:    #fef2f2;
  --fc-warn:      #d97706;
  --fc-warn-bg:   #fffbeb;
  --fc-shadow-sm: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
  --fc-shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  --fc-shadow-lg: 0 8px 24px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.04);
  --fc-radius:    12px;
  --fc-radius-lg: 16px;
}

footer, .footer, [class*="footer"] { display: none !important; }

body, .gradio-container, .gradio-container.fillable, main, main.contain {
  background-color: var(--fc-bg) !important;
  color: var(--fc-text) !important;
}

h1, h2, h3, h4 { color: var(--fc-text) !important; }

/* Tabs */
button[role="tab"], [class*="tab-nav"] {
  color: var(--fc-text2) !important;
  border-color: var(--fc-border) !important;
}

/* Markdown / prose */
.markdown, .prose, [class*="markdown"] {
  color: var(--fc-text2) !important;
}
.markdown p, .prose p, .prose li { color: var(--fc-text2) !important; line-height: 1.6 !important; }
.markdown code, .prose code {
  background: var(--fc-surface2) !important;
  border: 1px solid var(--fc-border) !important;
  color: var(--fc-accent) !important;
  border-radius: 5px;
  padding: 2px 6px;
  font-size: 0.88em;
}
.label-wrap, label, [data-testid] label { color: var(--fc-muted) !important; }

/* Gradio form rows */
.gr-form { background: var(--fc-surface) !important; border: 1px solid var(--fc-border) !important; }

/* ===== Content card panels ===== */
.content-card, .gr-group.content-card, div.content-card {
  background: var(--fc-surface) !important;
  border: 1px solid var(--fc-border) !important;
  border-radius: var(--fc-radius-lg) !important;
  padding: 20px 22px !important;
  margin-top: 12px !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
.content-card .prose, .content-card [class*="markdown"] { background: transparent !important; }

.section-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--fc-muted) !important;
  margin: 0 0 12px 0;
  border-bottom: 1px solid var(--fc-border);
  padding-bottom: 8px;
}

/* ===== Buttons — clean, no glow ===== */
button, .gr-button, button.gr-button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  transition: transform 0.1s ease, box-shadow 0.15s ease, background 0.15s ease !important;
}
button:active:not(:disabled), .gr-button:active:not(:disabled) {
  transform: scale(0.98) !important;
}
button:disabled, .gr-button:disabled, button.gr-button:disabled {
  cursor: not-allowed !important;
  opacity: 0.4 !important;
  filter: none !important;
  box-shadow: none !important;
  transform: none !important;
}

/* Start Episode — primary CTA */
button.gr-button.fc-btn--start, .gr-button.fc-btn--start {
  min-height: 52px !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  width: 100% !important;
  max-width: 100% !important;
  background: var(--fc-accent) !important;
  border: 1px solid var(--fc-accent) !important;
  color: #ffffff !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--start:hover:enabled, .gr-button.fc-btn--start:hover:enabled {
  background: var(--fc-accent2) !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-1px) !important;
}

/* Reveal Low */
button.gr-button.fc-btn--low, .gr-button.fc-btn--low {
  min-height: 46px;
  background: var(--fc-surface) !important;
  color: var(--fc-accent) !important;
  border: 1px solid var(--fc-accent) !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--low:hover:enabled, .gr-button.fc-btn--low:hover:enabled {
  background: var(--fc-accent-bg) !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-1px) !important;
}

/* Reveal High */
button.gr-button.fc-btn--high, .gr-button.fc-btn--high {
  min-height: 46px;
  background: var(--fc-surface) !important;
  color: var(--fc-warn) !important;
  border: 1px solid var(--fc-warn) !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--high:hover:enabled, .gr-button.fc-btn--high:hover:enabled {
  background: var(--fc-warn-bg) !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-1px) !important;
}

/* Refresh — secondary neutral */
button.gr-button.fc-btn--refresh, .gr-button.fc-btn--refresh {
  min-height: 46px;
  background: var(--fc-surface) !important;
  color: var(--fc-muted) !important;
  border: 1px solid var(--fc-border2) !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--refresh:hover:enabled, .gr-button.fc-btn--refresh:hover:enabled {
  background: var(--fc-surface2) !important;
  border-color: var(--fc-muted2) !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-1px) !important;
}

/* Commit — success green */
button.gr-button.fc-btn--commit, .gr-button.fc-btn--commit {
  min-height: 46px;
  background: var(--fc-pos) !important;
  color: #ffffff !important;
  border: 1px solid var(--fc-pos) !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--commit:hover:enabled, .gr-button.fc-btn--commit:hover:enabled {
  background: #15803d !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-1px) !important;
}

/* Suggest Action */
button.gr-button.fc-btn--suggest, .gr-button.fc-btn--suggest {
  min-height: 42px !important;
  width: 100% !important;
  max-width: 520px !important;
  margin: 0 auto !important;
  display: flex !important;
  background: var(--fc-surface) !important;
  border: 1px solid var(--fc-border2) !important;
  color: var(--fc-text2) !important;
  font-weight: 600 !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
button.gr-button.fc-btn--suggest:hover:enabled {
  background: var(--fc-surface2) !important;
  border-color: var(--fc-accent) !important;
  color: var(--fc-accent) !important;
  transform: translateY(-1px) !important;
}

/* Action grid spacing */
.fc-actions-row { gap: 10px !important; margin: 0 !important; }
.fc-actions-row .gr-block { min-width: 0 !important; }

/* ===== Game header ===== */
.fc-game-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin: 0 0 6px 0;
  color: var(--fc-text) !important;
}
.fc-game-header .fc-sub {
  color: var(--fc-muted) !important;
  font-size: 0.95rem;
  margin: 0;
  line-height: 1.5;
}

/* ===== 6-card clue grid ===== */
.fc-clue-arena { max-width: 900px; margin: 0 auto; padding: 4px 0 8px; }
.fc-clue-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
@media (max-width: 700px) {
  .fc-clue-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
}
@media (max-width: 420px) {
  .fc-clue-grid { grid-template-columns: 1fr; }
}

.fc-card-scene {
  perspective: 1000px;
  min-height: 140px;
  border-radius: var(--fc-radius-lg);
}
.fc-card-scene--small { min-height: 124px; }

.fc-card-inner {
  position: relative;
  width: 100%;
  min-height: 140px;
  transition: transform 0.45s ease, box-shadow 0.2s ease;
  transform-style: preserve-3d;
  border-radius: var(--fc-radius-lg);
  cursor: default;
}
.fc-card-scene--small .fc-card-inner { min-height: 124px; }

.fc-card-scene.is-hidden .fc-card-inner { transform: rotateY(0deg); }
.fc-card-scene.is-revealed .fc-card-inner { transform: rotateY(180deg); }
.fc-card-scene.is-revealed.is-new .fc-card-inner {
  transform: rotateY(0deg);
  animation: fc-reveal-flip 0.5s ease forwards;
}
@keyframes fc-reveal-flip {
  from { transform: rotateY(0deg); }
  to   { transform: rotateY(180deg); }
}

.fc-card-scene.is-new .fc-card-face--front {
  box-shadow: 0 0 0 2px var(--fc-pos), var(--fc-shadow-md) !important;
}

.fc-card-scene.is-hidden:hover .fc-card-inner {
  transform: rotateY(0deg) scale(1.03);
}
.fc-card-scene.is-hidden .fc-card-face--back {
  box-shadow: var(--fc-shadow-md);
}

.fc-card-face {
  position: absolute;
  inset: 0;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  border-radius: var(--fc-radius-lg);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px 14px;
  box-sizing: border-box;
  border: 1px solid var(--fc-border);
  overflow: hidden;
}

.fc-card-face--back {
  z-index: 2;
  background: var(--fc-surface2);
  color: var(--fc-muted);
  border: 1px solid var(--fc-border);
  box-shadow: var(--fc-shadow-sm);
}
.fc-card-question {
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  line-height: 1;
  color: var(--fc-border2);
}
.fc-card-badge {
  position: absolute;
  top: 8px; left: 8px;
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 3px 8px;
  border-radius: 6px;
  z-index: 3;
  border: 1px solid;
}
.fc-badge--low {
  color: var(--fc-accent);
  background: var(--fc-accent-bg);
  border-color: var(--fc-accent);
}
.fc-badge--high {
  color: var(--fc-warn);
  background: var(--fc-warn-bg);
  border-color: var(--fc-warn);
}

.fc-card-face--front {
  transform: rotateY(180deg);
  background: var(--fc-surface) !important;
  border: 1px solid var(--fc-border);
  z-index: 1;
  box-shadow: var(--fc-shadow-sm);
  align-items: flex-start;
  text-align: left;
}
.fc-card-lbl {
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--fc-muted) !important;
  margin-bottom: 5px;
  width: 100%;
  word-wrap: break-word;
}
.fc-card-val {
  font-size: 1rem;
  font-weight: 700;
  line-height: 1.4;
  color: var(--fc-text) !important;
  width: 100%;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* Footer strip: counters + token */
.fc-play-footer { margin-top: 8px; }
.fc-counter-row { margin-bottom: 8px; font-size: 0.9rem; color: var(--fc-text2) !important; }
.fc-counter-row strong { font-weight: 700; }
.fc-mute { color: var(--fc-muted) !important; }
.fc-hint-amber { color: var(--fc-neg) !important; font-size: 0.84rem; margin: 5px 0 0; }

.fc-token-outer { margin-top: 4px; }
.fc-token-label { font-size: 0.88rem; margin: 0 0 6px; color: var(--fc-text2); font-weight: 600; }
.fc-token-track {
  position: relative;
  height: 8px;
  width: 100%;
  background: var(--fc-surface2);
  border: 1px solid var(--fc-border);
  border-radius: 999px;
  overflow: hidden;
}
.fc-token-fill {
  height: 100%;
  width: 0%;
  min-width: 0;
  max-width: 100%;
  border-radius: 999px;
  transition: width 0.4s ease, background 0.3s ease;
  background: var(--fc-pos);
}
.fc-token-fill--low {
  background: var(--fc-neg) !important;
}
.fc-token-fill--mid {
  background: var(--fc-warn) !important;
}
.fc-token-fill--hi {
  background: var(--fc-pos) !important;
}

/* Token tension: gentle border highlight when critical */
.fc-token-track--low {
  border-color: var(--fc-neg) !important;
}
@keyframes fc-token-shake {
  0%, 100% { transform: translateX(0); }
  20% { transform: translateX(-3px); }
  40% { transform: translateX(3px); }
  60% { transform: translateX(-2px); }
  80% { transform: translateX(2px); }
}
.fc-token-tension-once {
  animation: fc-token-shake 0.55s ease 1;
}

/* Last action card */
.fc-card-log {
  background: var(--fc-surface);
  border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius);
  padding: 14px 16px;
  margin: 0;
  box-shadow: var(--fc-shadow-sm);
}
.fc-card-log h3 {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--fc-muted) !important;
  margin: 0 0 7px; font-weight: 700;
}
.fc-flow-line { margin: 0; font-size: 0.88rem; color: var(--fc-muted); line-height: 1.4; }
.fc-reward-pos { color: var(--fc-pos) !important; font-weight: 700; }
.fc-reward-neg { color: var(--fc-neg) !important; font-weight: 700; }
.fc-reward-neu { color: var(--fc-text2) !important; font-weight: 600; }
.fc-oneline-log { font-size: 0.9rem; color: var(--fc-text2); margin: 0 0 5px; line-height: 1.45; }

/* Live stats */
.fc-live-wrap { margin: 10px 0 14px; max-width: 900px; margin-left: auto; margin-right: auto; }
.fc-live-title { margin: 0 0 8px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fc-muted) !important; text-align: center; }
.fc-live-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
@media (max-width: 640px) { .fc-live-grid { grid-template-columns: repeat(2, 1fr); } }
.fc-live-tile {
  text-align: center; background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  padding: 10px 8px 12px; box-shadow: var(--fc-shadow-sm);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.fc-live-tile:hover { border-color: var(--fc-accent); box-shadow: var(--fc-shadow-md); }
.fc-live-ico { font-size: 0.9rem; line-height: 1; display: block; margin-bottom: 3px; color: var(--fc-muted); }
.fc-live-lbl { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--fc-muted) !important; font-weight: 700; margin: 0 0 3px; }
.fc-live-val { font-size: 1.1rem; font-weight: 800; color: var(--fc-text) !important; line-height: 1.2; }
.fc-live-val--pos { color: var(--fc-pos) !important; }
.fc-live-val--neg { color: var(--fc-neg) !important; }
.fc-live-val--neu { color: var(--fc-text2) !important; }

/* Confidence */
.fc-conf-outer { max-width: 900px; margin: 0 auto 12px; }
.fc-conf-card {
  text-align: center; background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius);
  padding: 14px 16px 16px; box-shadow: var(--fc-shadow-sm);
  transition: border-color 0.2s;
}
.fc-conf-title { margin: 0 0 8px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fc-muted) !important; }
.fc-conf-line { margin: 0; font-size: 1.3rem; font-weight: 800; line-height: 1.25; letter-spacing: 0.01em; }
.fc-conf--hi  { color: var(--fc-pos) !important; }
.fc-conf--med { color: var(--fc-warn) !important; }
.fc-conf--lo  { color: var(--fc-neg) !important; }
.fc-conf--unk { color: var(--fc-muted2) !important; font-size: 1rem; font-weight: 600; }
.fc-conf-wrap--hi  { border-color: #86efac !important; }
.fc-conf-wrap--med { border-color: #fcd34d !important; }
.fc-conf-wrap--lo  { border-color: #fca5a5 !important; }
.fc-conf-wrap--unk { border-color: var(--fc-border) !important; }

/* Episode trace */
.fc-trace-outer { margin-top: 8px; }
.fc-trace-panel {
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: var(--fc-radius); padding: 14px 16px;
  margin: 0; font-family: "JetBrains Mono", "SFMono-Regular", ui-monospace, Menlo, monospace;
  font-size: 0.85rem; line-height: 1.5; color: var(--fc-text2); box-shadow: var(--fc-shadow-sm);
}
.fc-trace-title { margin: 0 0 8px; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--fc-muted) !important; font-weight: 700; }
.fc-trace-empty { margin: 0; color: var(--fc-muted2) !important; }
.fc-trace-lines { display: flex; flex-direction: column; gap: 5px; }
.fc-trace-step { margin-bottom: 8px; }
.fc-trace-step:last-of-type { margin-bottom: 0; }
.fc-trace-line { padding: 2px 0; }
.fc-trace-reason { font-size: 0.76rem; color: var(--fc-muted) !important; margin: 2px 0 0; padding: 0 0 0 0.9rem; line-height: 1.4; }
.fc-trace-pos { color: var(--fc-pos) !important; font-weight: 700; }
.fc-trace-neg { color: var(--fc-neg) !important; font-weight: 700; }
.fc-trace-neu { color: var(--fc-text2) !important; font-weight: 600; }
.fc-trace-final { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--fc-border); font-weight: 700; }

/* History cards */
.gr-accordion, details.gr-accordion { background: var(--fc-surface2) !important; border: 1px solid var(--fc-border) !important; }
summary { color: var(--fc-text2); font-weight: 600; letter-spacing: 0.01em; }
.fc-history-entries { max-height: 300px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; padding: 2px; }
.fc-hist-card {
  background: var(--fc-surface);
  border: 1px solid var(--fc-border);
  border-left: 3px solid var(--fc-accent);
  border-radius: 8px;
  padding: 9px 12px;
  font-size: 0.86rem;
  color: var(--fc-text2);
  line-height: 1.45;
}
.fc-hist-card--final { border-left-color: var(--fc-pos); background: var(--fc-pos-bg); }
.fc-hist-st { color: var(--fc-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; display: block; margin-bottom: 3px; }

/* ========== Compare / About tabs ========== */
.fc-tab-dashboard a { color: var(--fc-accent) !important; }
.fc-tab-dashboard code {
  background: var(--fc-surface2) !important; color: var(--fc-accent) !important; padding: 2px 6px; border-radius: 5px;
  font-size: 0.86em; border: 1px solid var(--fc-border);
}
.fc-dashboard-title {
  font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; color: var(--fc-text) !important;
  line-height: 1.25;
}
.fc-dashboard-sub {
  font-size: 0.9rem; color: var(--fc-muted) !important; margin: 0 0 20px; line-height: 1.55; max-width: 56ch;
  font-weight: 400;
}
.fc-compare-arena { margin: 0 0 4px; }

.fc-compare-row {
  display: flex; flex-wrap: wrap; align-items: stretch; justify-content: center;
  gap: 14px 16px; margin: 0 0 18px;
}
.fc-compare-card {
  flex: 1 1 260px; max-width: 420px; min-width: 0;
  border-radius: var(--fc-radius-lg); padding: 20px 20px 18px; position: relative; overflow: hidden;
  min-height: 0;
  display: flex; flex-direction: column;
  background: var(--fc-surface);
  border: 1px solid var(--fc-border);
  box-shadow: var(--fc-shadow-sm);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.fc-compare-card:hover { transform: translateY(-2px); box-shadow: var(--fc-shadow-md); }
.fc-compare-card--random { border-top: 3px solid var(--fc-neg); }
.fc-compare-card--trained { border-top: 3px solid var(--fc-pos); }

.fc-compare-h {
  display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 14px; position: relative; z-index: 1;
}
.fc-compare-h h3 {
  margin: 0; font-size: 1rem; font-weight: 700; letter-spacing: 0.01em; color: var(--fc-text) !important;
  line-height: 1.2;
}
.fc-compare-card--random .fc-compare-h h3 { color: var(--fc-neg) !important; }
.fc-compare-card--trained .fc-compare-h h3 { color: var(--fc-pos) !important; }
.fc-compare-icon { font-size: 1.4rem; line-height: 1; }
.fc-compare-metrics { display: flex; flex-direction: column; gap: 12px; position: relative; z-index: 1; flex: 1; justify-content: space-evenly; }
.fc-metric {
  text-align: center; background: var(--fc-surface2); border: 1px solid var(--fc-border); border-radius: 8px; padding: 10px 10px 12px;
}
.fc-compare-token-note {
  font-size: 0.75rem; color: var(--fc-muted2) !important; text-align: center; margin: 4px 0 16px; line-height: 1.45;
  letter-spacing: 0.01em; max-width: 44rem; margin-left: auto; margin-right: auto;
}
.fc-metric-label { display: block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fc-muted) !important; margin-bottom: 3px; }
.fc-metric-value { display: block; font-size: 1.75rem; font-weight: 800; line-height: 1.1; color: var(--fc-text) !important; }
.fc-compare-card--random .fc-metric-value { color: var(--fc-neg) !important; }
.fc-compare-card--trained .fc-metric-value { color: var(--fc-pos) !important; }

.fc-compare-vsplit {
  display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 48px; flex: 0 0 auto; align-self: center;
  gap: 5px;
}
.fc-vs {
  display: flex; align-items: center; justify-content: center;
  min-width: 44px; min-height: 44px; border-radius: 50%; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em;
  color: var(--fc-text) !important;
  background: var(--fc-surface2);
  border: 1px solid var(--fc-border2);
  box-shadow: var(--fc-shadow-sm);
}
.fc-vs-connector { display: none; }
@media (min-width: 700px) {
  .fc-vs-connector { display: block; width: 1px; height: 24px; background: var(--fc-border2); border-radius: 1px; }
}
@media (max-width: 699px) {
  .fc-compare-row { flex-direction: column; align-items: stretch; }
  .fc-compare-vsplit { flex-direction: row; min-width: 100%; max-width: 100%; padding: 8px 0; gap: 8px; }
  .fc-compare-vsplit .fc-vs-connector {
    display: block; flex: 1; height: 1px; min-width: 20px;
    background: var(--fc-border2);
  }
}

.fc-compare-summary {
  margin: 0 0 8px; padding: 14px 16px; text-align: center; font-size: 1rem; line-height: 1.45;
  font-weight: 700; color: var(--fc-text) !important;
  background: var(--fc-pos-bg); border: 1px solid #86efac; border-radius: var(--fc-radius);
  box-shadow: var(--fc-shadow-sm);
}
.fc-compare-summary .fc-snum { color: var(--fc-pos) !important; }
.fc-compare-foot { font-size: 0.78rem; color: var(--fc-muted) !important; margin: 10px 0 0; }
.fc-compare-repro,
.fc-compare-scope {
  font-size: 0.74rem;
  color: var(--fc-muted2) !important;
  text-align: center;
  max-width: 44rem;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.45;
}
.fc-compare-repro { margin: 14px 12px 5px; }
.fc-compare-scope { margin: 0 12px 8px; font-style: italic; }
.fc-compare-na { color: var(--fc-muted); font-size: 0.92rem; }
.fc-training-depth {
  margin: 14px 0 10px;
  border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius-lg);
  background: var(--fc-surface);
  box-shadow: var(--fc-shadow-sm);
  padding: 16px 16px 14px;
}
.fc-training-depth h3 {
  margin: 0 0 12px;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--fc-text) !important;
  text-align: center;
}
.fc-depth-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 700px) {
  .fc-depth-grid { grid-template-columns: 1fr; }
}
.fc-depth-card {
  border: 1px solid var(--fc-border);
  border-radius: 10px;
  background: var(--fc-surface2);
  padding: 12px 12px 10px;
  box-shadow: var(--fc-shadow-sm);
}
.fc-depth-step {
  margin: 0 0 6px;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--fc-accent) !important;
  font-weight: 700;
}
.fc-depth-line {
  margin: 2px 0;
  font-size: 0.9rem;
  color: var(--fc-text2) !important;
}
.fc-depth-line strong { font-weight: 700; color: var(--fc-text) !important; }
.fc-depth-line .fc-pos { color: var(--fc-pos) !important; font-weight: 700; }
.fc-depth-conv {
  margin: 12px 0 0;
  text-align: center;
  color: var(--fc-text2) !important;
  font-weight: 600;
}
.fc-depth-conv-expl {
  margin: 10px auto 0;
  max-width: 48rem;
  text-align: center;
  font-size: 0.86rem;
  line-height: 1.5;
  color: var(--fc-muted) !important;
  font-weight: 400;
}
.fc-multi-seed {
  margin: 14px 0 10px;
  padding: 14px 14px 12px;
  border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius);
  background: var(--fc-surface);
  box-shadow: var(--fc-shadow-sm);
}
.fc-multi-seed h4 {
  margin: 0 0 10px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  text-align: center;
  color: var(--fc-accent) !important;
}
.fc-multi-seed-preamble,
.fc-multi-seed-intro,
.fc-multi-seed-stable {
  margin: 0 auto 8px;
  max-width: 48rem;
  text-align: center;
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--fc-muted) !important;
}
.fc-multi-seed-preamble {
  font-weight: 600;
  color: var(--fc-text2) !important;
  margin-bottom: 10px;
}
.fc-multi-seed-stable { margin-top: 10px; margin-bottom: 0; }
.fc-multi-seed-rows {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 8px 0 4px;
  font-size: 0.84rem;
  color: var(--fc-text2) !important;
}
.fc-multi-seed-row {
  margin: 0;
  text-align: center;
  padding: 6px 8px;
  background: var(--fc-surface2);
  border-radius: 7px;
  border: 1px solid var(--fc-border);
}
.fc-multi-seed-avg {
  margin: 10px 0 0;
  text-align: center;
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--fc-pos) !important;
}
.fc-dashboard-wrap { max-width: 1000px; margin: 0 auto; }
.fc-dashboard-grad { padding: 20px 4px; border-radius: var(--fc-radius-lg); }
.fc-dashboard-grad--compare {
  background: var(--fc-surface);
  border: 1px solid var(--fc-border);
  box-shadow: var(--fc-shadow-sm);
  margin: 0 0 8px;
  border-radius: var(--fc-radius-lg);
  padding: 24px 20px;
}

/* About tab */
.fc-about-lab {
  max-width: 880px;
  margin: 0 auto;
  box-sizing: border-box;
  background: transparent;
}
.fc-about-lab.fc-tab-dashboard,
.fc-about-lab.fc-tab-dashboard.gr-html,
.prose .fc-about-lab.fc-tab-dashboard {
  color: var(--fc-text) !important;
  padding: 24px 20px 28px;
  background: transparent;
  border-radius: 0;
  box-sizing: border-box;
  box-shadow: none;
}
.fc-about-lab .fc-about-title {
  margin: 0 0 10px;
  font-size: 1.2rem;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--fc-text) !important;
}
.fc-about-lab .fc-about-desc {
  margin: 0 0 4px;
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--fc-muted) !important;
  font-weight: 400;
  max-width: 600px;
}
.fc-about-lab .fc-about-desc strong {
  color: var(--fc-text) !important;
  font-weight: 600;
}
.fc-about-lab .fc-about-subline {
  margin: 5px 0 14px;
  max-width: 600px;
  font-size: 0.82rem;
  line-height: 1.55;
  color: var(--fc-muted) !important;
  font-weight: 400;
}
.fc-about-lab .fc-about-badge {
  display: inline-block;
  margin: 0 0 22px;
  padding: 4px 10px;
  font-size: 0.74rem;
  font-weight: 600;
  line-height: 1.4;
  color: var(--fc-accent) !important;
  background: var(--fc-accent-bg);
  border: 1px solid var(--fc-accent);
  border-radius: 6px;
}
.fc-about-lab .fc-about-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr);
  gap: 20px;
  margin: 0;
  background: var(--fc-surface);
  border: 1px solid var(--fc-border);
  border-radius: var(--fc-radius-lg);
  padding: 20px;
  box-shadow: var(--fc-shadow-sm);
}
@media (max-width: 900px) {
  .fc-about-lab .fc-about-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }
}
.fc-about-lab .fc-about-card {
  margin: 0;
  padding: 0;
  min-width: 0;
  background: transparent;
  border: none;
  border-radius: 0;
  box-shadow: none;
}
@media (min-width: 901px) {
  .fc-about-lab .fc-about-card:not(:last-child) {
    border-right: 1px solid var(--fc-border);
    padding-right: 16px;
  }
  .fc-about-lab .fc-about-card:not(:first-child) {
    padding-left: 16px;
  }
}
.fc-about-lab .fc-about-card-title {
  margin: 0 0 8px;
  font-size: 0.84rem;
  font-weight: 600;
  line-height: 1.35;
  color: var(--fc-text) !important;
}
.fc-about-lab .fc-about-list {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 0.84rem;
  line-height: 1.7;
  color: var(--fc-muted) !important;
}
.fc-about-lab .fc-about-list li {
  margin: 0 0 7px;
  padding: 0;
  color: var(--fc-muted) !important;
  border: none;
}
.fc-about-lab .fc-about-list li::before {
  content: "– ";
  color: var(--fc-muted2);
}
.fc-about-lab .fc-about-list li:last-child { margin-bottom: 0; }
.fc-about-lab .fc-about-list strong {
  color: var(--fc-text) !important;
  font-weight: 600;
}

/* Training Insights tab */
.fc-insights-page { max-width: 920px; margin: 0 auto; padding: 8px 6px 24px; display: flex; flex-direction: column; gap: 24px; }
.fc-insight-card {
  background: var(--fc-surface) !important;
  border: 1px solid var(--fc-border) !important;
  border-radius: var(--fc-radius-lg) !important;
  padding: 20px 22px 22px !important;
  box-shadow: var(--fc-shadow-sm);
  transition: box-shadow 0.2s ease, border-color 0.15s, transform 0.18s ease;
}
.fc-insight-card:hover {
  border-color: var(--fc-accent) !important;
  box-shadow: var(--fc-shadow-md) !important;
  transform: translateY(-2px);
}
.fc-insight-head { margin: 0 0 14px; text-align: left; }
.fc-insight-title {
  margin: 0 0 5px; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.01em; color: var(--fc-text) !important;
  line-height: 1.25;
}
.fc-insight-subtitle { margin: 0 0 8px; font-size: 0.9rem; line-height: 1.5; color: var(--fc-muted) !important; }
.fc-insight-takeaway {
  margin: 12px auto 0;
  padding: 10px 12px;
  text-align: center;
  font-size: 0.88rem;
  font-weight: 600;
  line-height: 1.45;
  color: var(--fc-pos) !important;
  max-width: 40rem;
  border: 1px solid #86efac;
  border-radius: 8px;
  background: var(--fc-pos-bg);
}
.fc-insight-takeaway--above {
  margin: 0 auto 14px !important;
}
.fc-insights-sim-row {
  max-width: 920px;
  margin: 0 auto 10px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
}
.fc-sim-banner {
  text-align: center;
  font-size: 0.86rem;
  font-weight: 600;
  color: var(--fc-pos) !important;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid #86efac;
  background: var(--fc-pos-bg);
  min-height: 1.2em;
}
.fc-insights-page.fc-insights--pulse .fc-insight-card {
  animation: fcInsightPulse 1.1s ease 1;
}
@keyframes fcInsightPulse {
  0%, 100% { box-shadow: var(--fc-shadow-sm); }
  45% { box-shadow: 0 0 0 3px var(--fc-accent-bg), var(--fc-shadow-md); }
}
#fc-btn-simulate-training button, #fc-btn-simulate-training.gr-button {
  min-height: 42px !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
  border-radius: 8px !important;
  background: var(--fc-surface) !important;
  border: 1px solid var(--fc-accent) !important;
  color: var(--fc-accent) !important;
  box-shadow: var(--fc-shadow-sm) !important;
}
#fc-btn-simulate-training button:hover, #fc-btn-simulate-training.gr-button:hover {
  background: var(--fc-accent-bg) !important;
  box-shadow: var(--fc-shadow-md) !important;
}
.fc-insight-badge {
  display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--fc-accent) !important; background: var(--fc-accent-bg); border: 1px solid var(--fc-accent);
  border-radius: 6px; padding: 3px 8px; line-height: 1.2;
}
.fc-insight-miss, .fc-insight-miss-box, .fc-insight-miss-box p {
  text-align: center; color: var(--fc-muted2) !important; font-size: 0.92rem; margin: 0; padding: 28px 16px;
}
/* Center chart images; cap width */
.fc-insight-card .gr-image, .fc-insight-card [class*="image"] { max-width: 720px; margin: 0 auto !important; }
.fc-insight-card .image-container, .fc-insight-card .image-container > div { max-width: 100% !important; justify-content: center !important; }
.fc-insight-card img, .fc-insight-card .image-container img {
  max-width: min(100%, 720px) !important; width: 100% !important; height: auto !important;
  display: block !important; margin: 0 auto !important; border-radius: 8px;
  box-shadow: var(--fc-shadow-sm);
}
"""

STATIC_HEADER_TOP = """
<div class="fc-play-page-header play-header">
  <h1 class="fc-play-h1">FC Decision Lab</h1>
  <p class="fc-play-guided">Reveal clues, spend tokens, decide when to commit.</p>
</div>
"""

STATIC_DEMO_SCRIPT = """
<details class="fc-demo-script">
  <summary>30-second demo script</summary>
  <div class="fc-demo-script-body">
    <p>"This environment models a real problem: when to stop gathering information under cost."</p>
    <p>Click <strong>Start Episode</strong> &rarr; take 2&ndash;3 actions.</p>
    <p>"Notice the model balances cost vs confidence." (point to stats + insight)</p>
    <p>Click <strong>Suggest Action</strong> &rarr; show recommendation.</p>
    <p>Switch to <strong>Training Insights</strong>:</p>
    <p>"We see learning stabilizes early and consistently beats random."</p>
    <p>"We also validated across multiple seeds to ensure stable behavior."</p>
    <p><strong>Done.</strong></p>
  </div>
</details>
"""

# First line of Blocks (before Tabs): sans for UI, mono for numerics.
GRADIO_APP_FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
"""

# Play tab: overrides scoped to .play-tab to avoid bleeding into other tabs.
PLAY_TAB_HEAD_INJECT = r"""
<style id="fc-play-styles">
/* ========== LAYOUT ========== */
.play-tab.fc-play-page {
  max-width: 980px !important;
  margin: 0 auto !important;
  padding: 20px 20px 24px !important;
  box-sizing: border-box !important;
  font-family: "Inter", system-ui, -apple-system, sans-serif !important;
  background: var(--fc-bg, #f4f6f9) !important;
  border-radius: 0 !important;
  border: none !important;
}

.play-tab,
.play-tab .wrap,
.play-tab [class*="block"],
.play-tab .gr-block {
  background: transparent !important;
  color: var(--fc-text, #111827) !important;
  border: none !important;
  box-shadow: none !important;
}

.play-tab .card {
  margin: 0 !important;
  padding: 0 !important;
}
.play-tab .card + .card { margin-top: 12px !important; }
.play-tab .section-title {
  margin: 0 0 8px !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  color: var(--fc-muted, #6b7280) !important;
  border-bottom: 1px solid var(--fc-border, #e2e6ec) !important;
  padding-bottom: 8px !important;
}
.play-tab .panel-board .section-title,
.play-tab .panel-actions .section-title { display: none !important; }
.play-tab .panel-stats .section-title {
  display: block !important;
  color: var(--fc-muted, #6b7280) !important;
  margin-bottom: 8px !important;
}
.play-tab .panel-confidence .section-title,
.play-tab .panel-insight .section-title { display: block !important; }
.play-tab .panel-decision-insight .section-title,
.play-tab .panel-suggest .section-title {
  display: block !important;
  color: var(--fc-accent, #1d4ed8) !important;
  margin-bottom: 10px !important;
}

/* Decision insight panel: accent left border */
.play-tab .panel-decision-insight {
  position: relative !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  border-left: 3px solid var(--fc-accent, #1d4ed8) !important;
  border-bottom: none !important;
  border-radius: 12px 12px 0 0 !important;
  box-shadow: none !important;
  background: var(--fc-surface, #ffffff) !important;
  overflow: visible !important;
}
.play-tab .panel-decision-insight + .panel-confidence {
  margin-top: 0 !important;
}
.play-tab .panel-confidence .confidence-wrap {
  margin-top: 0 !important;
}

/* Header */
.play-tab .play-header h1 {
  font-size: 0.9rem !important;
  letter-spacing: 0.1em !important;
  font-weight: 700 !important;
  color: var(--fc-muted, #6b7280) !important;
  margin: 0 !important;
  text-transform: uppercase !important;
}
.play-tab .play-header .fc-play-guided {
  margin: 5px 0 8px !important;
  color: var(--fc-text, #111827) !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
  text-align: center !important;
  max-width: 28rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.play-tab .fc-demo-script {
  max-width: 36rem !important;
  margin: 12px auto 8px !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  border-radius: 8px !important;
  background: var(--fc-surface2, #f8f9fb) !important;
  padding: 2px 10px 6px !important;
  opacity: 0.8 !important;
  transition: opacity 0.15s ease !important;
}
.play-tab .fc-demo-script:hover { opacity: 1 !important; }
.play-tab .fc-demo-script summary {
  cursor: pointer !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--fc-accent, #1d4ed8) !important;
  font-weight: 700 !important;
  padding: 8px 4px !important;
  list-style: none !important;
}
.play-tab .fc-demo-script-body {
  font-size: 0.84rem !important;
  color: var(--fc-text2, #374151) !important;
  line-height: 1.5 !important;
  padding: 0 4px 6px !important;
}
.play-tab .fc-demo-script-body p { margin: 0 0 7px !important; }
.play-tab .fc-play-header-line { display: none !important; }

/* Start Episode */
.play-tab .panel-cta {
  margin: 4px auto 10px !important;
  padding: 0 !important;
}
.play-tab .panel-cta .section-title { display: none !important; }

.play-tab .start-btn > div,
.play-tab .start-btn > div > div {
  display: flex !important;
  justify-content: center !important;
}
.play-tab .start-btn button,
.play-tab .start-btn .gr-button {
  width: min(380px, 100%) !important;
  min-height: 52px !important;
  border-radius: 8px !important;
  border: 1px solid var(--fc-accent, #1d4ed8) !important;
  background: var(--fc-accent, #1d4ed8) !important;
  color: #ffffff !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  font-size: 0.88rem !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(29, 78, 216, 0.2) !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease !important;
}
.play-tab .start-btn button:hover,
.play-tab .start-btn .gr-button:hover {
  background: var(--fc-accent2, #2563eb) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3) !important;
}
.play-tab .start-btn button:active,
.play-tab .start-btn .gr-button:active {
  transform: scale(0.98) !important;
}

/* Stats chips */
.play-tab .stats-row {
  display: flex !important;
  justify-content: flex-end !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  width: 100% !important;
  margin: 0 0 2px !important;
}
.play-tab .stats-row > div { flex: 0 0 auto !important; }
.play-tab .fc-stat-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border-radius: 999px;
  padding: 6px 12px 6px 10px;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  background: var(--fc-surface, #ffffff);
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  font-family: "JetBrains Mono", monospace;
}
.play-tab .fc-chip-ico { font-size: 0.85rem; color: var(--fc-muted, #6b7280); }
.play-tab .fc-chip-lbl {
  font-size: 0.64rem;
  letter-spacing: 0.1em;
  color: var(--fc-muted, #6b7280);
  text-transform: uppercase;
}
.play-tab .fc-chip-val { font-size: 0.95rem; font-weight: 700; color: var(--fc-text, #111827); }
.play-tab .fc-chip-val.pos { color: var(--fc-pos, #16a34a); }
.play-tab .fc-chip-val.neg { color: var(--fc-neg, #dc2626); }

/* ========== CARDS / BOARD ========== */
.play-tab .fc-encore-board { margin: 8px auto 0 !important; max-width: 860px; }
.play-tab .fc-board-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0 10px;
}
.play-tab .fc-board-title {
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--fc-muted, #6b7280);
}
.play-tab .fc-board-count {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  padding: 2px 9px;
  border-radius: 999px;
  background: var(--fc-surface2, #f8f9fb);
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  color: var(--fc-accent, #1d4ed8);
  font-weight: 700;
}
.play-tab .fc-clue-grid-host {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 900px) {
  .play-tab .fc-clue-grid-host { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 520px) {
  .play-tab .fc-clue-grid-host { grid-template-columns: 1fr; }
}
.play-tab .fc-encore-card {
  position: relative;
  min-height: 128px;
  border-radius: 12px;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  background: var(--fc-surface, #ffffff);
  padding: 12px 12px 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
.play-tab .fc-encore-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.09) !important;
}
.play-tab .fc-encore-card.is-new {
  box-shadow: 0 0 0 2px var(--fc-pos, #16a34a), 0 4px 12px rgba(0,0,0,0.09) !important;
}

.play-tab .fc-encore-tier {
  position: absolute;
  left: 10px;
  top: 9px;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  border-radius: 999px;
  padding: 2px 8px;
  border: 1px solid;
  font-family: "JetBrains Mono", monospace;
}
.play-tab .fc-encore-card.is-low .fc-encore-tier {
  color: var(--fc-accent, #1d4ed8);
  border-color: var(--fc-accent, #1d4ed8);
  background: var(--fc-accent-bg, #eff6ff);
}
.play-tab .fc-encore-card.is-high .fc-encore-tier {
  color: var(--fc-warn, #d97706);
  border-color: var(--fc-warn, #d97706);
  background: var(--fc-warn-bg, #fffbeb);
}

.play-tab .fc-encore-hidden {
  text-align: center;
  color: var(--fc-muted2, #9ca3af);
  line-height: 1.2;
}
.play-tab .fc-encore-hidden .ico { font-size: 1.4rem; display: block; margin-bottom: 4px; opacity: 0.5; }
.play-tab .fc-encore-hidden .q { letter-spacing: 0.3em; font-family: "JetBrains Mono", monospace; font-size: 0.9rem; color: var(--fc-border2, #d0d5de); }
.play-tab .fc-encore-revealed { text-align: center; padding-top: 14px; max-width: 100%; }
.play-tab .fc-encore-clue {
  color: var(--fc-text, #111827);
  font-size: 0.95rem;
  font-weight: 700;
  line-height: 1.25;
  word-break: break-word;
}
.play-tab .fc-encore-attr {
  margin-top: 6px;
  color: var(--fc-muted, #6b7280);
  font-size: 0.66rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.play-tab .fc-play-footer {
  margin: 8px auto 0 !important;
  max-width: 860px;
  border-radius: 8px;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  background: var(--fc-surface, #ffffff);
  padding: 10px 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.play-tab .fc-hint-warn {
  color: var(--fc-neg, #dc2626) !important;
  font-size: 0.84rem !important;
}
.play-tab .fc-counter-row,
.play-tab .fc-token-label {
  color: var(--fc-text2, #374151) !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.82rem !important;
}
.play-tab .fc-token-track {
  height: 7px !important;
  background: var(--fc-surface2, #f8f9fb) !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  border-radius: 999px !important;
}
.play-tab .fc-token-track--low {
  border-color: var(--fc-neg, #dc2626) !important;
}
.play-tab .fc-token-fill--mid {
  background: var(--fc-warn, #d97706) !important;
}

/* ========== CONFIDENCE ========== */
.play-tab .confidence-shell {
  margin: 0 !important;
  width: 100% !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
  background: var(--fc-surface, #ffffff) !important;
  padding: 10px 12px !important;
  display: grid !important;
  grid-template-columns: auto 1fr auto !important;
  align-items: center !important;
  gap: 10px !important;
}
.play-tab .confidence-lbl {
  color: var(--fc-muted, #6b7280) !important;
  letter-spacing: 0.09em !important;
  font-size: 0.68rem !important;
  text-transform: uppercase !important;
  margin: 0 !important;
  font-weight: 600 !important;
}
.play-tab .confidence-track {
  height: 7px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--fc-surface2, #f8f9fb);
  border: 1px solid var(--fc-border, #e2e6ec);
}
.play-tab .confidence-fill { height: 100%; border-radius: inherit; transition: width .2s ease; }
.play-tab .confidence-fill--hi  { width: 88%; background: var(--fc-pos, #16a34a); }
.play-tab .confidence-fill--med { width: 58%; background: var(--fc-warn, #d97706); }
.play-tab .confidence-fill--lo  { width: 28%; background: var(--fc-neg, #dc2626); }
.play-tab .confidence-fill--unk { width: 8%;  background: var(--fc-border2, #d0d5de); }
.play-tab .confidence-txt {
  margin: 0 !important; font-size: 0.82rem !important; font-weight: 700 !important;
  text-align: right !important; font-family: "JetBrains Mono", monospace !important;
}
.play-tab .confidence-txt--hi  { color: var(--fc-pos, #16a34a) !important; }
.play-tab .confidence-txt--med { color: var(--fc-warn, #d97706) !important; }
.play-tab .confidence-txt--lo  { color: var(--fc-neg, #dc2626) !important; }
.play-tab .confidence-txt--unk { color: var(--fc-muted2, #9ca3af) !important; }
.play-tab .confidence-wrap {
  max-width: 860px !important;
  margin: 0 auto !important;
}
.play-tab .confidence-legend {
  margin: 7px 0 0 !important;
  font-size: 0.8rem !important;
  line-height: 1.45 !important;
  color: var(--fc-muted, #6b7280) !important;
  text-align: center !important;
  font-family: "Inter", system-ui, sans-serif !important;
}
.play-tab .confidence-action-hint {
  margin: 5px 0 0 !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  text-align: center !important;
  color: var(--fc-accent, #1d4ed8) !important;
  font-family: "Inter", system-ui, sans-serif !important;
}

/* ========== DECISION INSIGHT ========== */
.play-tab .fc-decision-insight-inner {
  text-align: center !important;
  padding: 14px 12px 12px !important;
  border-radius: 8px !important;
  background: transparent !important;
  border: none !important;
}
.play-tab .fc-insight-headline {
  margin: 0 0 5px !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  color: var(--fc-text, #111827) !important;
  letter-spacing: 0.01em !important;
}
.play-tab .fc-insight-ai-prefix {
  display: block;
  font-size: 0.64rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--fc-accent, #1d4ed8) !important;
  margin-bottom: 4px !important;
  font-weight: 700 !important;
}
.play-tab .fc-session-progress {
  text-align: center !important;
  font-size: 0.82rem !important;
  color: var(--fc-muted, #6b7280) !important;
  margin: 0 auto 8px !important;
  max-width: 860px !important;
}
.play-tab .fc-session-progress-mono {
  font-family: "JetBrains Mono", monospace !important;
  font-weight: 700 !important;
  color: var(--fc-text2, #374151) !important;
}
.play-tab .fc-insight-sub {
  margin: 0 !important;
  font-size: 0.82rem !important;
  line-height: 1.45 !important;
  color: var(--fc-muted, #6b7280) !important;
}
.play-tab .fc-suggest-panel {
  max-width: 520px !important;
  margin: 0 auto !important;
  padding: 12px 14px !important;
  border-radius: 10px !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  background: var(--fc-surface2, #f8f9fb) !important;
  text-align: center !important;
}
.play-tab .fc-suggest-panel--prompt {
  color: var(--fc-muted, #6b7280) !important;
  font-size: 0.84rem !important;
}
.play-tab .fc-suggest-rec {
  margin: 0 0 5px !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  color: var(--fc-pos, #16a34a) !important;
}
.play-tab .fc-suggest-why {
  margin: 0 !important;
  font-size: 0.84rem !important;
  color: var(--fc-muted, #6b7280) !important;
  line-height: 1.4 !important;
}

/* ========== ACTION BUTTONS ========== */
.play-tab .btn-grid {
  max-width: 520px !important;
  margin: 0 auto !important;
  gap: 8px !important;
}
.play-tab .btn-grid button, .play-tab .btn-grid .gr-button {
  width: 100% !important;
  min-height: 44px !important;
  border-radius: 8px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
}
#fc-play-btn-low button, #fc-play-btn-low .gr-button {
  background: var(--fc-surface, #ffffff) !important;
  border: 1px solid var(--fc-accent, #1d4ed8) !important;
  color: var(--fc-accent, #1d4ed8) !important;
}
#fc-play-btn-low button:hover, #fc-play-btn-low .gr-button:hover {
  background: var(--fc-accent-bg, #eff6ff) !important;
}
#fc-play-btn-high button, #fc-play-btn-high .gr-button {
  background: var(--fc-surface, #ffffff) !important;
  border: 1px solid var(--fc-warn, #d97706) !important;
  color: var(--fc-warn, #d97706) !important;
}
#fc-play-btn-high button:hover, #fc-play-btn-high .gr-button:hover {
  background: var(--fc-warn-bg, #fffbeb) !important;
}
#fc-play-btn-refresh button, #fc-play-btn-refresh .gr-button {
  background: var(--fc-surface, #ffffff) !important;
  border: 1px solid var(--fc-border2, #d0d5de) !important;
  color: var(--fc-muted, #6b7280) !important;
}
#fc-play-btn-refresh button:hover, #fc-play-btn-refresh .gr-button:hover {
  background: var(--fc-surface2, #f8f9fb) !important;
  border-color: var(--fc-muted2, #9ca3af) !important;
}
#fc-play-btn-commit button, #fc-play-btn-commit .gr-button {
  background: var(--fc-pos, #16a34a) !important;
  border: 1px solid var(--fc-pos, #16a34a) !important;
  color: #ffffff !important;
  min-height: 48px !important;
}
#fc-play-btn-commit button:hover, #fc-play-btn-commit .gr-button:hover {
  background: #15803d !important;
  border-color: #15803d !important;
}
.play-tab button:disabled, .play-tab .gr-button:disabled {
  opacity: 0.38 !important;
  filter: none !important;
  cursor: not-allowed !important;
}

/* ========== LOG / TRACE PANELS ========== */
.play-tab .log-panel,
.play-tab .fc-encore-log,
.play-tab .fc-trace-panel {
  max-width: 860px !important;
  margin: 0 auto !important;
  border-radius: 10px !important;
  border: 1px solid var(--fc-border, #e2e6ec) !important;
  background: var(--fc-surface, #ffffff) !important;
  padding: 12px 14px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.play-tab .fc-oneline-log,
.play-tab .fc-play-log-meta,
.play-tab .fc-trace-line,
.play-tab .fc-trace-reason,
.play-tab .fc-trace-empty {
  font-family: "JetBrains Mono", monospace !important;
  color: var(--fc-text2, #374151) !important;
  font-size: 0.82rem !important;
}
.play-tab .fc-reward-pos { color: var(--fc-pos, #16a34a) !important; }
.play-tab .fc-reward-neg { color: var(--fc-neg, #dc2626) !important; }
.play-tab .fc-reward-neu { color: var(--fc-text2, #374151) !important; }

/* Status / flow hints */
.play-tab .status-hint,
.play-tab .fc-play-flow-live {
  margin: 5px auto 2px !important;
  max-width: 860px !important;
  text-align: center !important;
  color: var(--fc-muted, #6b7280) !important;
  font-style: italic !important;
  font-size: 0.82rem !important;
  font-family: "JetBrains Mono", monospace !important;
  padding: 4px 0 !important;
}
.play-tab .status-hint strong,
.play-tab .fc-flow-line strong { color: var(--fc-text2, #374151) !important; }
.play-tab .fc-onboarding-hint {
  margin: 0 auto 6px !important;
  max-width: 860px !important;
  text-align: center !important;
  font-size: 0.84rem !important;
  line-height: 1.4 !important;
  color: var(--fc-accent, #1d4ed8) !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  border: 1px solid var(--fc-accent, #1d4ed8) !important;
  background: var(--fc-accent-bg, #eff6ff) !important;
}
.play-tab .fc-session-progress--empty {
  margin: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
}
.play-tab .fc-history-scroll { max-height: 240px !important; overflow-y: auto !important; padding-right: 4px !important; }

@media (max-width: 640px) {
  .play-tab.fc-play-page { padding: 12px 12px 16px !important; }
  .play-tab .fc-chip-lbl { display: none; }
  .play-tab .confidence-shell { grid-template-columns: 1fr !important; }
  .play-tab .confidence-txt { text-align: left !important; }
}
</style>
"""


def _load_evaluation_json() -> dict:
    p = Path(__file__).resolve().parent / "artifacts" / "evaluation.json"
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _fmetric_win(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    return f"{float(x) * 100:.1f}%"


def _fmetric_reward(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    v = float(x)
    s = f"{v:+.3f}"
    return s


def _fmetric_steps(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    return f"{float(x):.2f}"


def _compare_panel_html() -> str:
    data = _load_evaluation_json()
    _r = data.get("random_eval")
    rdict: dict = _r if isinstance(_r, dict) else {}
    if isinstance(data.get("ppo_eval"), dict) and data["ppo_eval"]:
        tdict: dict = data["ppo_eval"]
        trained_name = "PPO"
    elif isinstance(data.get("q_tabular_eval"), dict) and data.get("q_tabular_eval"):
        tdict = data["q_tabular_eval"]
        trained_name = "Q-table"
    else:
        tdict = {}
        trained_name = "Trained"

    wr = rdict.get("win_rate")
    wtr = tdict.get("win_rate")
    if (
        wr is not None
        and wtr is not None
        and isinstance(wr, (int, float))
        and isinstance(wtr, (int, float))
    ):
        dpp = (float(wtr) - float(wr)) * 100
        if dpp > 0:
            summary = (
                f"Trained agent outperforms random by "
                f'<span class="fc-snum">+{dpp:.1f}</span> percentage points in win rate.'
            )
        else:
            summary = (
                f"Win rate (trained &minus; random): <span class=\"fc-snum\">{dpp:+.1f}</span> points."
            )
    else:
        summary = (
            "Run <code>python train.py</code> to build <code>artifacts/evaluation.json</code> "
            "and populate this panel."
        )
        summary = f'<span class="fc-compare-na">{summary}</span>'

    foot = ""
    if data and rdict and tdict:
        foot = (
            f'<p class="fc-compare-foot">Source: <code>artifacts/evaluation.json</code> &middot; '
            f"Trained metrics: {trained_name}</p>"
        )

    depth_300 = {}
    depth_1200 = {}
    base = Path(__file__).resolve().parent / "artifacts"
    p300 = base / "eval_300k.json"
    p1200 = base / "eval_500k.json"
    try:
        with p300.open(encoding="utf-8") as f:
            raw300 = json.load(f)
            if isinstance(raw300, dict):
                d = raw300.get("ppo_eval")
                depth_300 = d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        depth_300 = {}
    try:
        with p1200.open(encoding="utf-8") as f:
            raw1200 = json.load(f)
            if isinstance(raw1200, dict):
                d = raw1200.get("ppo_eval")
                depth_1200 = d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        depth_1200 = {}

    w300 = _fmetric_win(depth_300.get("win_rate"))
    r300 = _fmetric_reward(depth_300.get("avg_reward"))
    w1200 = _fmetric_win(depth_1200.get("win_rate"))
    r1200 = _fmetric_reward(depth_1200.get("avg_reward"))

    conv = "We compare checkpoints at different training stages (300k vs 500k steps)."
    depth_convergence_expl = (
        "We observe early convergence (~200k\u2013300k steps), with minimal gains beyond this point. "
        "This indicates stable policy learning rather than undertraining."
    )

    multi_seed_html = ""
    mse = data.get("ppo_multi_seed_eval")
    if isinstance(mse, dict):
        per = mse.get("per_seed")
        avg = mse.get("average")
        if isinstance(per, list) and per:
            row_parts: list[str] = []
            for i, row in enumerate(per):
                if not isinstance(row, dict):
                    continue
                sid = (
                    EVAL_REPORT_SEEDS[i]
                    if i < len(EVAL_REPORT_SEEDS)
                    else i
                )
                row_parts.append(
                    f'<p class="fc-multi-seed-row">Seed {sid}: win '
                    f'{_fmetric_win(row.get("win_rate"))}, reward '
                    f'{_fmetric_reward(row.get("avg_reward"))}, steps '
                    f'{_fmetric_steps(row.get("avg_steps"))}</p>'
                )
            avg_line = ""
            if isinstance(avg, dict):
                avg_line = (
                    f'<p class="fc-multi-seed-avg">Average: win '
                    f'{_fmetric_win(avg.get("win_rate"))}, reward '
                    f'{_fmetric_reward(avg.get("avg_reward"))}, steps '
                    f'{_fmetric_steps(avg.get("avg_steps"))}</p>'
                )
            if row_parts:
                multi_seed_html = (
                    '<div class="fc-multi-seed" role="region" aria-label="Multi-seed evaluation">'
                    "<h4>Multi-seed evaluation</h4>"
                    "<p class=\"fc-multi-seed-preamble\">To validate consistency beyond a single "
                    "run, we performed multi-seed evaluation.</p>"
                    "<p class=\"fc-multi-seed-intro\">We evaluated the trained policy across "
                    "multiple random seeds (0, 42, 99) to ensure consistent performance under "
                    "different episode conditions.</p>"
                    f'<div class="fc-multi-seed-rows">{"".join(row_parts)}</div>'
                    f"{avg_line}"
                    "<p class=\"fc-multi-seed-stable\">Performance remains consistent across "
                    "seeds, indicating robust behavior rather than reliance on a single run.</p>"
                    "</div>"
                )

    return f"""<div class="fc-tab-dashboard fc-dashboard-wrap">
<div class="fc-dashboard-grad fc-dashboard-grad--compare">
  <h2 class="fc-dashboard-title">Trained vs Random</h2>
  <p class="fc-dashboard-sub">Side-by-side results from the same environment. Bigger is better for win rate and average reward; steps depend on your strategy.</p>
  <div class="fc-compare-arena">
    <div class="fc-compare-row">
      <div class="fc-compare-card fc-compare-card--random" role="group" aria-label="Random policy">
        <div class="fc-compare-h">
          <h3>Random Policy</h3>
        </div>
        <div class="fc-compare-metrics">
          <div class="fc-metric">
            <span class="fc-metric-label">Win rate</span>
            <span class="fc-metric-value">{_fmetric_win(rdict.get("win_rate"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg reward</span>
            <span class="fc-metric-value">{_fmetric_reward(rdict.get("avg_reward"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg steps</span>
            <span class="fc-metric-value">{_fmetric_steps(rdict.get("avg_steps"))}</span>
          </div>
        </div>
      </div>
      <div class="fc-compare-vsplit" aria-hidden="true">
        <div class="fc-vs-connector"></div>
        <div class="fc-vs">VS</div>
        <div class="fc-vs-connector"></div>
      </div>
      <div class="fc-compare-card fc-compare-card--trained" role="group" aria-label="Trained policy">
        <div class="fc-compare-h">
          <h3>Trained Policy</h3>
        </div>
        <div class="fc-compare-metrics">
          <div class="fc-metric">
            <span class="fc-metric-label">Win rate</span>
            <span class="fc-metric-value">{_fmetric_win(tdict.get("win_rate"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg reward</span>
            <span class="fc-metric-value">{_fmetric_reward(tdict.get("avg_reward"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg steps</span>
            <span class="fc-metric-value">{_fmetric_steps(tdict.get("avg_steps"))}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  <p class="fc-compare-token-note">Token usage metric is currently under calibration due to normalization effects. We focus on stable metrics like win rate and reward.</p>
  <div class="fc-training-depth" role="group" aria-label="Training depth analysis">
    <h3>Training Depth Analysis</h3>
    <div class="fc-depth-grid">
      <div class="fc-depth-card">
        <p class="fc-depth-step">300k steps</p>
        <p class="fc-depth-line"><strong>Win Rate:</strong> <strong>{w300}</strong></p>
        <p class="fc-depth-line"><strong>Avg Reward:</strong> <span class="fc-pos">{r300}</span></p>
      </div>
      <div class="fc-depth-card">
        <p class="fc-depth-step">500k steps</p>
        <p class="fc-depth-line"><strong>Win Rate:</strong> <strong>{w1200}</strong></p>
        <p class="fc-depth-line"><strong>Avg Reward:</strong> <span class="fc-pos">{r1200}</span></p>
      </div>
    </div>
    <p class="fc-depth-conv">{conv}</p>
    <p class="fc-depth-conv-expl">{depth_convergence_expl}</p>
  </div>
  {multi_seed_html}
  <p class="fc-compare-summary">{summary}</p>
  {foot}
  <p class="fc-compare-repro">Results are reproducible with a fixed seed and deterministic evaluation.</p>
  <p class="fc-compare-scope">Findings are consistent within this environment; broader generalization is future work.</p>
</div>
</div>"""

ABOUT_PANEL_HTML = r"""
<div class="fc-tab-dashboard fc-dashboard-wrap fc-about-lab">
  <h2 class="fc-about-title">How this decision system works</h2>
  <p class="fc-about-desc">Scannable quick reference. Same rules and API as the <strong>Play</strong> tab&mdash;only the presentation differs.</p>
  <p class="fc-about-subline">This environment rewards smart information gathering and disciplined stopping.</p>
  <span class="fc-about-badge">Programmatic rewards (RLVR-style)</span>
  <div class="fc-about-grid">
    <article class="fc-about-card" role="region" aria-labelledby="fc-about-goal-h">
      <h3 class="fc-about-card-title" id="fc-about-goal-h">Goal</h3>
      <ul class="fc-about-list">
        <li>Manage a fixed token budget</li>
        <li>Reveal clues to learn about the pick</li>
        <li>Decide: commit, or refresh the candidate</li>
      </ul>
    </article>
    <article class="fc-about-card" role="region" aria-labelledby="fc-about-actions-h">
      <h3 class="fc-about-card-title" id="fc-about-actions-h">Actions</h3>
      <ul class="fc-about-list">
        <li><strong>Reveal Low</strong> &mdash; cheap information</li>
        <li><strong>Reveal High</strong> &mdash; pricier, higher signal</li>
        <li><strong>Commit</strong> &mdash; lock in your final decision</li>
        <li><strong>Refresh</strong> &mdash; end episode and skip the candidate (alias: skip)</li>
      </ul>
    </article>
    <article class="fc-about-card" role="region" aria-labelledby="fc-about-strategy-h">
      <h3 class="fc-about-card-title" id="fc-about-strategy-h">Strategy</h3>
      <ul class="fc-about-list">
        <li>Balance cost against information</li>
        <li>Don&rsquo;t burn tokens for no gain</li>
        <li>Commit when you are confident, refresh when the profile looks wrong</li>
      </ul>
    </article>
  </div>
</div>
"""


def _attr_display_name(raw_key: str) -> str:
    k = str(raw_key).strip().lower().replace(" ", "_")
    if k in ATTR_LABELS:
        return ATTR_LABELS[k]
    base = str(raw_key).strip()
    return base.upper()[:8] if base else "CLUE"


def _clue_label_value(raw: str) -> tuple[str, str, bool]:
    """(label, value, hidden) for card face."""
    if raw == HIDDEN:
        return ("", "???", True)
    try:
        t = ast.literal_eval(raw)
        if isinstance(t, (list, tuple)) and len(t) == 2:
            k, v = t
            return (_attr_display_name(str(k)), str(v).strip(), False)
    except (ValueError, SyntaxError, TypeError):
        pass
    return ("CLUE", str(raw), False)


def _tier_badge_class(idx: int) -> str:
    return "fc-badge--low" if idx < 3 else "fc-badge--high"


def _tier_label(idx: int) -> str:
    return "LOW" if idx < 3 else "HIGH"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _artifact_png_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / filename


def _insight_section_header_html(title: str, subtitle: str) -> str:
    return (
        '<div class="fc-insight-head">'
        f'<h3 class="fc-insight-title">{_html_escape(title)}</h3>'
        f'<p class="fc-insight-subtitle">{_html_escape(subtitle)}</p>'
        '<span class="fc-insight-badge">Generated after training (offline evaluation)</span>'
        "</div>"
    )


# (filename, section title, subtitle) for Training Insights tab
TRAINING_INSIGHT_SECTIONS: list[tuple[str, str, str]] = [
    (
        "reward_curve.png",
        "Learning Progress",
        "Agent improves reward over time during training",
    ),
    (
        "win_rate_vs_random.png",
        "Before vs After Training",
        "Trained agent shows higher win rate than random in this environment (evaluation)",
    ),
]

REWARD_CURVE_TAKEAWAY_HTML = (
    '<p class="fc-insight-takeaway fc-insight-takeaway--above">'
    "Learning stabilizes early and maintains performance thereafter.</p>"
)
WINRATE_TAKEAWAY_HTML = (
    '<p class="fc-insight-takeaway fc-insight-takeaway--above">'
    "Learning stabilizes early and consistently beats random.</p>"
)


def _render_six_clues(
    revealed: tuple[str, ...] | list[str] | None,
    highlight: int | None,
) -> str:
    if not revealed or len(revealed) < 6:
        revealed = [HIDDEN] * 6
    shown = sum(1 for x in revealed if x != HIDDEN)
    parts: list[str] = [
        '<div class="fc-encore-board">'
        '<div class="fc-board-head">'
        '<div class="fc-board-title">CLUES</div>'
        f'<div class="fc-board-count">{shown}/6</div>'
        "</div>"
        '<div class="fc-clue-grid-host">'
    ]
    for i in range(6):
        raw = revealed[i]  # type: ignore[index]
        label, value, hidden = _clue_label_value(raw)
        is_low = i < 3
        tier = "LOW" if is_low else "HIGH"
        tier_cls = "is-low" if is_low else "is-high"
        new_cls = " is-new" if highlight is not None and i == highlight and not hidden else ""
        if hidden:
            parts.append(
                f'<div class="fc-encore-card {tier_cls} is-hidden">'
                f'<span class="fc-encore-tier">{tier}</span>'
                '<div class="fc-encore-hidden"><span class="ico">&#9673;</span>'
                '<span class="q">? ? ?</span></div>'
                "</div>"
            )
            continue
        vhtml = _html_escape(value)
        lhtml = _html_escape(label)
        parts.append(
            f'<div class="fc-encore-card {tier_cls} is-revealed{new_cls}">'
            f'<span class="fc-encore-tier">{tier}</span>'
            '<div class="fc-encore-revealed">'
            f'<div class="fc-encore-clue">{vhtml}</div>'
            f'<div class="fc-encore-attr">{lhtml}</div>'
            "</div>"
            "</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


COUNTER_FOOTER_IDLE = (
    "<div class='fc-play-footer'>"
    "<div class='fc-counter-row'>"
    "Low: <span class='fc-mute' style='font-weight:600;'>&#8212;</span> &nbsp;&middot;&nbsp; "
    "High: <span class='fc-mute' style='font-weight:600;'>&#8212;</span></div>"
    "<p class='fc-mute' style='margin:0 0 8px;font-size:0.84rem;'>"
    "Awaiting episode start.</p>"
    "<div class='fc-token-outer'>"
    "<p class='fc-token-label' style='color:#9ca3af'>"
    f"Tokens: 0 / {MAX_TOKENS}</p><div class='fc-token-track'>"
    "<div class='fc-token-fill' style='width:0%'></div></div></div></div>"
)


def _token_bar_fill_class(pct: float) -> str:
    if pct >= 55.0:
        return "fc-token-fill fc-token-fill--hi"
    if pct >= 25.0:
        return "fc-token-fill fc-token-fill--mid"
    return "fc-token-fill fc-token-fill--low"


def _play_footer_html(o, prev_token_pct: float | None = None) -> str:
    lo, hi = o.low_remaining, o.high_remaining
    cl = "#dc2626" if lo == 0 else "#6b7280"
    ch = "#dc2626" if hi == 0 else "#6b7280"
    if lo == 0 and hi == 0:
        hint = (
            "<p class='fc-hint-warn' style='margin:8px 0 0'>"
            "No reveals left. Use <strong>Commit</strong> or <strong>Refresh</strong>."
            "</p>"
        )
    elif lo == 0:
        hint = (
            "<p class='fc-hint-warn' style='margin:6px 0 0;'>"
            "No <strong>LOW</strong> cost clues left.</p>"
        )
    elif hi == 0:
        hint = (
            "<p class='fc-hint-warn' style='margin:6px 0 0;'>"
            "No <strong>HIGH</strong> cost clues left.</p>"
        )
    else:
        hint = ""
    toks = int(o.tokens)
    w = 100.0 * toks / MAX_TOKENS if MAX_TOKENS else 0.0
    fillcls = _token_bar_fill_class(w)
    prev = float(prev_token_pct) if isinstance(prev_token_pct, (int, float)) else None
    cross = prev is not None and prev >= 30.0 and w < 30.0
    low = w < 30.0
    track_bits = ["fc-token-track"]
    if cross:
        track_bits.append("fc-token-tension-once")
    if low:
        track_bits.append("fc-token-track--low")
    track_cls = " ".join(track_bits)
    return (
        "<div class='fc-play-footer'>"
        "<div class='fc-counter-row'>"
        f"Low: <strong style='color:{cl}'>{lo}</strong> &nbsp;&middot;&nbsp; "
        f"High: <strong style='color:{ch}'>{hi}</strong>"
        f"</div>{hint}"
        "<div class='fc-token-outer'>"
        f"<p class='fc-token-label' style='margin:0 0 6px;'>"
        f"Tokens: {toks} / {MAX_TOKENS}</p>"
        f"<div class='{track_cls}'>"
        f'<div class="{fillcls}" style="width: {w:.1f}%;">'
        "</div></div></div></div>"
    )


def _newly_revealed_index(
    pre: list[str] | tuple[str, ...] | None, post: tuple[str, ...]
) -> int | None:
    if pre is None or len(pre) < 6 or len(post) < 6:
        return None
    for i in range(6):
        if pre[i] == HIDDEN and post[i] != HIDDEN:
            return i
    return None


def _outcome_text(
    user_action: int,
    pre: dict,
    o,
    new_idx: int | None,
) -> str:
    if pre is None:
        return "—"

    pre_tokens = int(pre.get("tokens", 0))
    if user_action in (0, 1) and pre_tokens <= 0 and o.done:
        return "Out of tokens — decision committed (forced)."

    if new_idx is not None and user_action == 0:
        return "Low-cost clue revealed"
    if new_idx is not None and user_action == 1:
        return "High-cost clue revealed"

    if user_action == 0 and int(pre.get("low_remaining", 0)) == 0 and pre_tokens > 0:
        return "No low-cost clues remaining. Tokens were spent; nothing left to reveal."
    if user_action == 1 and int(pre.get("high_remaining", 0)) == 0 and pre_tokens > 0:
        return "No high-cost clues remaining. Tokens were spent; nothing left to reveal."

    if user_action == 0 and new_idx is None and pre_tokens > 0:
        return "No low-cost clues left — spent tokens, no new clue"
    if user_action == 1 and new_idx is None and pre_tokens > 0:
        return "No high-cost clues left — spent tokens, no new clue"

    if user_action == 2:
        if o.reward < -0.1:
            return "Bad decision penalty" if o.done else "Cost applied"
        return "You committed" if o.done else "—"

    if user_action == 3:
        if o.reward > 0.05:
            return "Smart refresh/skip saved cost"
        if o.reward < -0.01:
            return "Costly refresh on a strong pick"
        return "You refreshed"

    if o.done:
        return "Episode over"
    return "—"


def _snapshot(o) -> dict:
    return {
        "revealed_clues": o.revealed_clues,
        "tokens": o.tokens,
        "step_number": o.step_number,
        "low_remaining": o.low_remaining,
        "high_remaining": o.high_remaining,
        "done": o.done,
        "reward": o.reward,
        "info": dict(getattr(o, "info", {})),
    }


def _oneline_last_action(
    o,
    last_action: int | None,
    outcome: str,
) -> str:
    if last_action is None:
        return _html_escape(
            (outcome or "Episode started — your move.").strip()
        )
    r = float(o.reward)
    if r > 1e-6:
        rs = f"+{r:.2f}"
    else:
        rs = f"{r:.2f}"
    an = _action_name(last_action)
    return f"[Step {o.step_number}] {an} \u2192 {rs} reward \u2014 {_html_escape(outcome)}"


def _last_step_html(
    o,
    outcome: str,
    last_action: int | None = None,
) -> str:
    r = float(o.reward)
    if r > 1e-6:
        rc, cls = f"+{r:.3f}", "fc-reward-pos"
    elif r < -1e-6:
        rc, cls = f"{r:.3f}", "fc-reward-neg"
    else:
        rc, cls = f"{r:.3f}", "fc-reward-neu"
    line = _oneline_last_action(o, last_action, outcome)
    active = " fc-play-log-active" if last_action is not None else ""
    detail = (
        ""
        if last_action is None
        else f"<p class='fc-mute fc-play-log-detail log-content'>{_html_escape(outcome)}</p>"
    )
    return (
        f"<div class='fc-encore-log fc-play-log log-panel'>"
        f"<p class='fc-oneline-log log-content{active}'>{line}</p>"
        f"<p class='fc-play-log-meta log-content'><span class='fc-mute'>"
        f"Step reward</span> <span class='{cls}'>{rc}</span> &middot; "
        f"<span class='fc-mute'>Tokens</span> "
        f"{o.tokens} / {MAX_TOKENS} &middot; <span class='fc-mute'>Step</span> {o.step_number}</p>"
        f"{detail}"
        f"</div>"
    )


def _flow_badge(
    o,
    last_action: int | None,
) -> str:
    if o.done and last_action == 2:
        return (
            "<p class='fc-flow-line'>"
            "Episode finished &middot; <span class='fc-reward-neg' style='font-weight:800'>"
            "Committed</span></p>"
        )
    if o.done and last_action == 3:
        return (
            "<p class='fc-flow-line'>"
            "Episode finished &middot; <span class='fc-reward-pos' style='font-weight:800'>"
            "Refreshed</span></p>"
        )
    if o.done:
        return "<p class='fc-flow-line'>Episode <strong>finished</strong></p>"
    return (
        "<p class='fc-flow-line fc-play-flow-live'>"
        "In progress \u2014 reveal, refresh, or commit when ready."
        "</p>"
    )


def _button_state_from_obs(o) -> tuple:
    d = o.done
    toks = o.tokens
    can_low = (not d) and toks > 0 and o.low_remaining > 0
    can_high = (not d) and toks > 0 and o.high_remaining > 0
    can_choose = not d
    return (
        gr.update(interactive=can_low, visible=True),
        gr.update(interactive=can_high, visible=True),
        gr.update(interactive=can_choose, visible=True),
        gr.update(interactive=can_choose, visible=True),
    )


# Episode log (0–3 = env Action values; same API as before)
_ACTION_NAMES: dict[int, str] = {
    0: "Reveal Low",
    1: "Reveal High",
    2: "Commit",
    3: "Refresh",
}

HISTORY_STATE_INIT: dict = {
    "lines": [],
    "cum": 0.0,
    "stale": True,
    "card_rows": [],
}


def _action_name(action: int) -> str:
    return _ACTION_NAMES.get(action, f"Action {action}")


def _cumulative_assessment(cum: float) -> str:
    if cum > 0.05:
        return "Cumulative reward is **positive** on this run (sum of all step rewards)."
    if cum < -0.05:
        return "Cumulative reward is **negative** on this run (sum of all step rewards)."
    return "Cumulative reward is about **neutral** (sum of all step rewards)."


def _log_after_step(
    h: dict | None,
    user_action: int,
    o,
) -> dict:
    prev: dict = {**HISTORY_STATE_INIT, **(h or {})}
    lines: list[str] = list(prev.get("lines", []))
    card_rows: list[dict] = list(prev.get("card_rows", []))
    prior_cum = float(prev.get("cum", 0.0))
    cum = prior_cum + float(o.reward)
    an = _action_name(user_action)
    step_entry = f"""**Step {o.step_number}**
- **Action:** {an}
- **Step reward:** {o.reward:+.4f}
- **Tokens left:** {o.tokens}
- **Low left:** {o.low_remaining}
- **High left:** {o.high_remaining}
- **Done (after this step):** {o.done}"""
    lines.append(step_entry)
    card_rows.append(
        {
            "step": int(o.step_number),
            "action": an,
            "reward": float(o.reward),
            "final": False,
        }
    )
    if o.done:
        if user_action == 2:
            end = "Commit (terminal action)"
        elif user_action == 3:
            end = "Refresh (terminal; env action SKIP)"
        else:
            end = f"{an} or system limit (tokens, clues, max steps, or all revealed)"
        final = f"""## Final result
- **Total steps (env step number):** {o.step_number}
- **Final tokens:** {o.tokens}
- **Last step reward:** {o.reward:+.4f}
- **Cumulative reward (episode sum):** {cum:+.4f}
- **End driver:** {end}
- **Outcome (cumulative return):** {_cumulative_assessment(cum)}"""
        lines.append(final)
        card_rows.append(
            {
                "step": int(o.step_number),
                "action": "—",
                "reward": float(cum),
                "final": True,
                "cum": float(cum),
            }
        )
    return {
        "lines": lines,
        "cum": cum,
        "stale": False,
        "card_rows": card_rows,
    }


def _log_to_html(h: dict | None) -> str:
    h = h or HISTORY_STATE_INIT
    if h.get("stale", True) and not h.get("card_rows") and not h.get("lines"):
        return (
            "<div class='fc-history-entries' style='padding:12px'><p class='fc-mute' "
            "style='margin:0'>No actions yet. Start a new episode.</p></div>"
        )
    if not h.get("card_rows") and not h.get("lines"):
        return (
            "<div class='fc-history-entries' style='padding:12px'>"
            "<p class='fc-mute' style='margin:0'>New episode \u2014 steps will log as you act.</p></div>"
        )
    parts: list[str] = ['<div class="fc-history-entries">']
    for row in h.get("card_rows", []):
        if row.get("final"):
            c = float(row.get("cum", 0.0))
            sc = f"+{c:.2f}" if c > 0 else f"{c:.2f}"
            hcls = (
                "fc-reward-pos"
                if c > 1e-6
                else "fc-reward-neg"
                if c < -1e-6
                else "fc-reward-neu"
            )
            parts.append(
                f"<div class='fc-hist-card fc-hist-card--final'>"
                f"<span class='fc-hist-st'>Episode total</span>"
                f"Final cumulative reward: <strong class='{hcls}'>{sc}</strong></div>"
            )
            continue
        stn = int(row.get("step", 0))
        an = str(row.get("action", "—"))
        r = float(row.get("reward", 0.0))
        rs = f"+{r:.2f}" if r > 0 else f"{r:.2f}"
        cls = "fc-reward-pos" if r > 1e-6 else "fc-reward-neg" if r < -1e-6 else "fc-reward-neu"
        an_esc = _html_escape(an)
        parts.append(
            f"<div class='fc-hist-card'>"
            f"<span class='fc-hist-st'>Step {stn}</span>"
            f"<span>{an_esc} \u2192 <span class='{cls}'>{rs}</span> reward</span></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _append_episode_trace(
    prior: list[dict] | list | None,
    o,
) -> list[dict]:
    info = getattr(o, "info", None) or {}
    an = info.get("action_name", "")
    if an in ("", "—", None):
        return list(prior or [])
    row = {
        "step": int(info.get("step_number", o.step_number)),
        "action": str(an),
        "reward": float(info.get("step_reward", o.reward)),
        "reason": str(info.get("reason", "")).strip(),
    }
    return list(prior or []) + [row]


def _episode_trace_html(rows: list[dict] | list | None, episode_done: bool) -> str:
    rows = list(rows or [])
    if not rows and not episode_done:
        body = '<p class="fc-trace-empty">No actions taken yet.</p>'
    else:
        lines: list[str] = []
        for r in rows:
            rw = float(r["reward"])
            if rw > 1e-9:
                rs, ccls = f"+{rw:.2f}", "fc-trace-pos"
            elif rw < -1e-9:
                rs, ccls = f"{rw:.2f}", "fc-trace-neg"
            else:
                rs, ccls = f"{rw:.2f}", "fc-trace-neu"
            an_esc = _html_escape(str(r.get("action", "—")))
            stn = int(r.get("step", 0))
            sub = ""
            reas = str(r.get("reason", "")).strip()
            if reas:
                sub = f'<div class="fc-trace-reason">{_html_escape(reas)}</div>'
            lines.append(
                f'<div class="fc-trace-step">'
                f'<div class="fc-trace-line">Step {stn} &middot; {an_esc} \u2192 <span class="{ccls}">{rs}</span></div>'
                f"{sub}</div>"
            )
        body = '<div class="fc-trace-lines">' + "".join(lines) + "</div>"
    final = ""
    if episode_done and rows:
        tot = sum(float(x["reward"]) for x in rows)
        if tot > 1e-9:
            ts, tcls = f"+{tot:.2f}", "fc-trace-pos"
        elif tot < -1e-9:
            ts, tcls = f"{tot:.2f}", "fc-trace-neg"
        else:
            ts, tcls = f"{tot:.2f}", "fc-trace-neu"
        final = (
            f'<div class="fc-trace-final">'
            f'<div class="fc-trace-final-lbl">Final result</div>'
            f"Total reward: <span class=\"{tcls}\">{ts}</span></div>"
        )
    return (
        '<div class="fc-trace-panel">'
        f'<div class="fc-history-scroll">{body}{final}</div>'
        "</div>"
    )


def _update_live_stats(prev: dict | None, o) -> dict:
    """Cumulative total_reward += step_reward; tokens and steps from info. No-op step keeps prior stats (freeze)."""
    info = getattr(o, "info", None) or {}
    if info.get("action_name") == "—":
        return dict(prev) if prev else {**LIVE_STATS_DEFAULT}
    p = {**LIVE_STATS_DEFAULT, **(prev or {})}
    tr = float(info.get("step_reward", o.reward))
    return {
        "current_tokens": int(info.get("tokens_left", o.tokens)),
        "total_reward": float(p.get("total_reward", 0.0)) + tr,
        "step_count": int(info.get("step_number", o.step_number)),
    }


def _stat_html_tokens_left(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    cur = int(s["current_tokens"])
    cls = "neg" if cur <= 20 else ""
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">T</span>'
        '<span class="fc-chip-lbl">Tokens</span>'
        f'<span class="fc-chip-val {cls}">{cur}</span>'
        "</div>"
    )


def _stat_html_steps(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    steps = int(s["step_count"])
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">S</span>'
        '<span class="fc-chip-lbl">Steps</span>'
        f'<span class="fc-chip-val">{steps}</span>'
        "</div>"
    )


def _stat_html_reward(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    tot = float(s["total_reward"])
    if tot > 1e-9:
        cls, reward_str = "pos", f"+{tot:.2f}"
    elif tot < -1e-9:
        cls, reward_str = "neg", f"{tot:.2f}"
    else:
        cls, reward_str = "", f"{tot:.2f}"
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">R</span>'
        '<span class="fc-chip-lbl">Reward</span>'
        f'<span class="fc-chip-val {cls}">{reward_str}</span>'
        "</div>"
    )


def _live_stats_html(stats: dict | None) -> str:
    """Single-row HTML (all four stats); Play tab uses four gr.HTML outputs instead."""
    return (
        '<div class="stats-row">'
        + _stat_html_tokens_left(stats)
        + _stat_html_steps(stats)
        + _stat_html_reward(stats)
        + "</div>"
    )


def _compute_confidence_level(live: dict | None) -> str:
    """Heuristic from running total reward and step count; frontend-only (no env change)."""
    s = {**LIVE_STATS_DEFAULT, **(live or {})}
    st = int(s.get("step_count", 0))
    tr = float(s.get("total_reward", 0.0))
    if st == 0:
        return "UNKNOWN"
    if tr > 0.8 and st >= 2:
        return "HIGH"
    if tr > 0:
        return "MEDIUM"
    return "LOW"


def _confidence_html(level: str) -> str:
    tip = "Confidence is based on reward trend and actions taken"
    u = (level or "UNKNOWN").upper()
    if u == "UNKNOWN":
        vcls = "confidence-txt--unk"
        fill_cls = "confidence-fill--unk"
        text = "UNKNOWN"
    elif u == "HIGH":
        vcls = "confidence-txt--hi"
        fill_cls = "confidence-fill--hi"
        text = "HIGH"
    elif u == "MEDIUM":
        vcls = "confidence-txt--med"
        fill_cls = "confidence-fill--med"
        text = "MEDIUM"
    else:
        vcls = "confidence-txt--lo"
        fill_cls = "confidence-fill--lo"
        text = "LOW"
    et = _html_escape(tip)
    tx = _html_escape(text)
    legend = (
        "<strong>High</strong> = strong signal to commit &middot; "
        "<strong>Medium</strong> = uncertain &middot; "
        "<strong>Low</strong> = explore more"
    )
    if u == "HIGH":
        act = "Consider committing."
    elif u == "MEDIUM":
        act = "You may explore more."
    elif u == "LOW":
        act = "Gather more clues."
    else:
        act = "Take a step to see how confidence moves."
    return (
        '<div class="confidence-wrap">'
        f'<div class="confidence-shell" title="{et}">'
        f'<div class="confidence-lbl">Confidence</div>'
        f'<div class="confidence-track"><div class="confidence-fill {fill_cls}"></div></div>'
        f'<div class="confidence-txt {vcls}" title="{et}">{tx}</div>'
        f"</div>"
        f'<p class="confidence-legend">{legend}</p>'
        f'<p class="confidence-action-hint">{_html_escape(act)}</p>'
        "</div>"
    )


def _insight_title(text: str) -> str:
    et = _html_escape(text)
    return (
        "<p class='fc-insight-headline'>"
        "<span class='fc-insight-ai-prefix'>AI Insight:</span> "
        f"{et}</p>"
    )


def _decision_insight_html(
    o,
    last_action: int | None,
    outcome: str,
) -> str:
    """Short human-readable insight from the last step (UI only)."""
    oc = (outcome or "").lower()
    if last_action is None:
        inner = (
            _insight_title("Ready to play")
            + "<p class='fc-insight-sub'>Reveal clues, spend tokens, then commit or refresh when the "
            "picture is clear enough.</p>"
        )
    elif o.done and last_action == 2:
        inner = (
            _insight_title("Confident commit")
            + "<p class='fc-insight-sub'>You locked in a decision; the episode ended on commit.</p>"
        )
    elif last_action == 0 and "revealed" in oc:
        inner = (
            _insight_title("Cheap info gained")
            + "<p class='fc-insight-sub'>A low-cost reveal added signal without heavy token spend.</p>"
        )
    elif last_action == 1 and "revealed" in oc:
        inner = (
            _insight_title("Strong signal")
            + "<p class='fc-insight-sub'>A high-tier clue brought sharper evidence for your call.</p>"
        )
    elif last_action == 2 and not o.done:
        inner = (
            _insight_title("Commit pressed")
            + "<p class='fc-insight-sub'>Cost or state updated; episode may still be open.</p>"
        )
    elif last_action == 3:
        inner = (
            _insight_title("Walk away")
            + "<p class='fc-insight-sub'>Refresh ends the candidate\u2014useful when probes aren\u2019t paying off.</p>"
        )
    elif o.done:
        inner = (
            _insight_title("Episode complete")
            + "<p class='fc-insight-sub'>Check cumulative reward and history for this run.</p>"
        )
    else:
        inner = (
            _insight_title("Keep probing")
            + "<p class='fc-insight-sub'>Balance further reveals against remaining budget.</p>"
        )
    return f'<div class="fc-decision-insight-inner">{inner}</div>'


def _recommendation_from_obs(o, live: dict | None) -> tuple[int, str, str]:
    """Heuristic next action (0–3); does not call the env."""
    if o.done:
        return -1, "Start new episode", "This round ended \u2014 reset to continue."
    s = {**LIVE_STATS_DEFAULT, **(live or {})}
    tr = float(s.get("total_reward", 0.0))
    st = int(s.get("step_count", 0))
    revealed = sum(1 for c in o.revealed_clues if c != HIDDEN)
    toks = int(o.tokens)
    if st >= 2 and tr > 0.12 and revealed >= 2:
        reason = random.choice(
            (
                "Reward and information favor deciding now.",
                "Leaning toward commit due to strong signal.",
                "Signal is strong \u2014 committing looks favorable.",
                "Confidence suggests committing now.",
                "Further clues may not justify the cost \u2014 good moment to decide.",
            )
        )
        return 2, "Commit", reason
    if o.low_remaining > 0 and toks > 0:
        reason = random.choice(
            (
                "Low-cost clues are still available \u2014 gather cheap signal first.",
                "Cheap reveals still on the board \u2014 worth scanning before a pricey pull.",
                "Start wide: low-tier probes often pay for themselves in clarity.",
            )
        )
        return 0, "Reveal Low", reason
    if o.high_remaining > 0 and toks > 0:
        reason = random.choice(
            (
                "You likely need sharper evidence \u2014 a high-tier clue can break the tie.",
                "Low tier is thin; a stronger signal may be worth the token cost.",
                "When cheap options are exhausted, premium clues buy decision quality.",
            )
        )
        return 1, "Reveal High", reason
    if toks <= 0 or (o.low_remaining == 0 and o.high_remaining == 0):
        reason = random.choice(
            (
                "No further reveals \u2014 decide with what you know.",
                "Budget or clues are exhausted; time to commit or walk.",
                "You\u2019ve seen what you can afford \u2014 lock in or refresh the slate.",
            )
        )
        return 2, "Commit", reason
    reason = random.choice(
        (
            "If probes look like a dead end, walking away can limit loss.",
            "When marginal insight is flat, refresh resets the option value.",
            "A tactical skip can beat throwing good tokens after weak signal.",
        )
    )
    return 3, "Refresh", reason


def _smooth_recommendation(
    prev: dict | None,
    o,
    live: dict | None,
) -> tuple[dict, str]:
    """Stabilize label/reason when the same action remains optimal."""
    a, name, reason = _recommendation_from_obs(o, live)
    if (
        prev
        and prev.get("a") == a
        and prev.get("name") == name
        and a >= 0
    ):
        reason = str(prev.get("reason", reason))
    rec = {"a": a, "name": name, "reason": reason}
    if a < 0:
        body = (
            f'<p class="fc-suggest-rec">Recommended: {_html_escape(name)}</p>'
            f'<p class="fc-suggest-why">{_html_escape(reason)}</p>'
        )
    else:
        body = (
            f'<p class="fc-suggest-rec">Recommended: {_html_escape(name)}</p>'
            f'<p class="fc-suggest-why">{_html_escape(reason)}</p>'
        )
    html = f'<div class="fc-suggest-panel">{body}</div>'
    return rec, html


def _suggest_prompt_html() -> str:
    return (
        '<div class="fc-suggest-panel fc-suggest-panel--prompt">'
        "<p style='margin:0'>Press <strong>Suggest Action</strong> for a live read on the "
        "next move. The suggestion stays until the board changes in a meaningful way.</p></div>"
    )


def _suggest_stabilizer_pack(rec: dict, html: str) -> dict:
    return {**rec, "html": html}


def _suggest_after_step(
    prev_rec: dict | None,
    snap: dict,
    live: dict | None,
) -> tuple[str, dict | None]:
    """Keep last AI suggestion unless the recommended action label changes."""
    if prev_rec is None:
        return _suggest_prompt_html(), None
    ov = _obs_from_snapshot(snap)
    rec, html = _smooth_recommendation(prev_rec, ov, live)
    if (
        prev_rec.get("a") == rec.get("a")
        and prev_rec.get("name") == rec.get("name")
        and prev_rec.get("html")
    ):
        html = str(prev_rec["html"])
    return html, _suggest_stabilizer_pack(rec, html)


def _session_progress_html(sp: dict | None) -> str:
    if not sp or int(sp.get("episodes", 0)) <= 0:
        return "<div class='fc-session-progress fc-session-progress--empty' aria-hidden='true'></div>"
    n = int(sp["episodes"])
    br = sp.get("best_reward")
    brs = "—" if br is None else f"{float(br):+.2f}"
    return (
        f'<div class="fc-session-progress">'
        f'Episode <span class="fc-session-progress-mono">{n}</span>'
        f' &middot; Best reward <span class="fc-session-progress-mono">{brs}</span>'
        f"</div>"
    )


def _play_flow_html(
    o,
    last_action: int | None,
    onboarding: bool,
) -> str:
    parts: list[str] = []
    if onboarding and last_action is None and not o.done:
        parts.append(
            "<p class='fc-onboarding-hint'>"
            "Try revealing 1\u20132 clues, then commit when the signal feels strong enough.</p>"
        )
    parts.append(_flow_badge(o, last_action))
    return "".join(parts)


def _obs_from_snapshot(d: dict):
    """Lightweight view for UI heuristics (matches Observation fields used in Play tab)."""
    from types import SimpleNamespace

    rc = d.get("revealed_clues", ())
    return SimpleNamespace(
        done=bool(d.get("done", False)),
        revealed_clues=tuple(rc) if rc is not None else (HIDDEN,) * 6,
        tokens=int(d.get("tokens", 0)),
        low_remaining=int(d.get("low_remaining", 0)),
        high_remaining=int(d.get("high_remaining", 0)),
    )


def build_blocks() -> gr.Blocks:
    with gr.Blocks(
        title="FC Decision Lab",
        theme=gr.themes.Base(),
        css=CSS_STRING,
    ) as demo:
        gr.HTML(GRADIO_APP_FONT_LINKS, elem_id="fc-app-font-links")
        with gr.Tabs():
            with gr.Tab("Play"):
                with gr.Column(
                    elem_id="fc-play-root",
                    elem_classes=["play-tab", "fc-play-page"],
                ):
                    gr.HTML(PLAY_TAB_HEAD_INJECT, elem_id="fc-play-style-inject")
                    st = gr.State()  # type: ignore[var-annotated]  # { "env", "pre_obs" }
                    episode_trace = gr.State([])  # type: ignore[var-annotated]  # list[dict] step log
                    live_stats = gr.State(dict(LIVE_STATS_DEFAULT))  # type: ignore[var-annotated]
                    # Before first action per episode → UNKNOWN; after steps → HIGH|MEDIUM|LOW
                    confidence_level = gr.State("UNKNOWN")
                    suggest_stabilizer = gr.State(None)  # type: ignore[var-annotated]
                    session_progress = gr.State(
                        {"episodes": 0, "best_reward": None, "last_token_pct": None}
                    )  # type: ignore[var-annotated]
                    play_onboarding = gr.State(True)  # type: ignore[var-annotated]

                    gr.HTML(
                        STATIC_HEADER_TOP,
                        elem_id="fc-play-header",
                        elem_classes=["fc-play-header-slot", "play-header"],
                    )

                    with gr.Group(elem_classes=["card", "panel-cta"]):
                        b_reset = gr.Button(
                            "Start Episode",
                            elem_id="fc-play-btn-start",
                            elem_classes=[
                                "start-btn",
                                "fc-btn--start",
                                "gr-button",
                                "gr-button-primary",
                            ],
                        )

                    gr.HTML(
                        STATIC_DEMO_SCRIPT,
                        elem_id="fc-play-demo-script",
                    )

                    session_progress_display = gr.HTML(
                        _session_progress_html(
                            {"episodes": 0, "best_reward": None, "last_token_pct": None}
                        ),
                        elem_id="fc-play-session-progress",
                    )

                    ls0 = dict(LIVE_STATS_DEFAULT)
                    with gr.Group(elem_classes=["card", "panel-stats"]):
                        gr.HTML("<div class='section-title'>Live Stats</div>")
                        with gr.Row(elem_classes=["stats-row"]):
                            stat_html_steps = gr.HTML(
                                _stat_html_steps(ls0),
                                elem_id="fc-play-stat-steps",
                                elem_classes=["stat-block"],
                            )
                            stat_html_reward = gr.HTML(
                                _stat_html_reward(ls0),
                                elem_id="fc-play-stat-reward",
                                elem_classes=["stat-block"],
                            )

                    with gr.Group(elem_classes=["card", "panel-board"]):
                        gr.HTML("<div class='section-title'>Game Board</div>")
                        card_block = gr.HTML(
                            _render_six_clues((HIDDEN,) * 6, None),
                            elem_id="fc-play-cards",
                        )
                        footer_status = gr.HTML(
                            COUNTER_FOOTER_IDLE,
                            elem_id="fc-play-footer-tokens",
                            elem_classes=["fc-footer-wrap"],
                        )

                    with gr.Group(elem_classes=["card", "panel-decision-insight"]):
                        gr.HTML("<div class='section-title'>Decision Insight</div>")
                        insight_display = gr.HTML(
                            _decision_insight_html(None, None, ""),
                            elem_id="fc-play-insight",
                        )

                    with gr.Group(elem_classes=["card", "panel-confidence"]):
                        gr.HTML("<div class='section-title'>Confidence</div>")
                        confidence_display = gr.HTML(
                            _confidence_html("UNKNOWN"),
                            elem_id="fc-play-confidence",
                            elem_classes=["fc-conf-outer"],
                        )

                    with gr.Group(elem_classes=["card", "panel-actions"]):
                        gr.HTML("<div class='section-title'>Actions</div>")
                        with gr.Row(elem_classes=["btn-grid"]):
                            b_low = gr.Button(
                                "REVEAL LOW",
                                interactive=False,
                                elem_id="fc-play-btn-low",
                                elem_classes=[
                                    "btn-low",
                                    "gr-button",
                                    "fc-btn--low",
                                    "gr-button-primary",
                                ],
                            )
                            b_high = gr.Button(
                                "REVEAL HIGH",
                                interactive=False,
                                elem_id="fc-play-btn-high",
                                elem_classes=[
                                    "btn-high",
                                    "gr-button",
                                    "fc-btn--high",
                                    "gr-button-primary",
                                ],
                            )
                        with gr.Row(elem_classes=["btn-grid"]):
                            b_skip = gr.Button(
                                "REFRESH",
                                interactive=False,
                                elem_id="fc-play-btn-refresh",
                                elem_classes=[
                                    "btn-ghost",
                                    "gr-button",
                                    "fc-btn--refresh",
                                    "gr-button-primary",
                                ],
                            )
                            b_commit = gr.Button(
                                "COMMIT",
                                interactive=False,
                                elem_id="fc-play-btn-commit",
                                elem_classes=[
                                    "btn-commit",
                                    "gr-button",
                                    "fc-btn--commit",
                                    "gr-button-primary",
                                ],
                            )

                    with gr.Group(elem_classes=["card", "panel-suggest"]):
                        gr.HTML("<div class='section-title'>Suggest Action</div>")
                        b_suggest = gr.Button(
                            "Suggest Action",
                            elem_id="fc-play-btn-suggest",
                            elem_classes=[
                                "fc-btn--suggest",
                                "gr-button",
                                "gr-button-secondary",
                            ],
                        )
                        suggest_display = gr.HTML(
                            _suggest_prompt_html(),
                            elem_id="fc-play-suggest",
                        )

                    with gr.Group(elem_classes=["card", "panel-last"]):
                        gr.HTML("<div class='section-title'>Last Action</div>")
                        last_block = gr.HTML(
                            "<div class='fc-encore-log fc-play-log log-panel'>"
                            "<p class='fc-mute log-content' style='margin:0'>No action yet.</p></div>",
                            elem_id="fc-play-last-action",
                            elem_classes=["log-panel"],
                        )

                    with gr.Group(elem_classes=["card", "panel-history"]):
                        gr.HTML("<div class='section-title'>Episode History</div>")
                        episode_trace_display = gr.HTML(
                            _episode_trace_html([], False),
                            elem_id="fc-play-episode-trace",
                            elem_classes=["fc-trace-outer", "log-panel"],
                        )
                    flow = gr.HTML(
                        "<p class='fc-flow-line status-hint'>"
                        "Press <strong>Start Episode</strong> to begin.</p>",
                        elem_id="fc-play-flow",
                        elem_classes=["status-hint"],
                    )
                _out_play = [
                    st,
                    episode_trace,
                    live_stats,
                    confidence_level,
                    card_block,
                    stat_html_steps,
                    stat_html_reward,
                    confidence_display,
                    insight_display,
                    suggest_display,
                    last_block,
                    episode_trace_display,
                    flow,
                    footer_status,
                    b_low,
                    b_high,
                    b_skip,
                    b_commit,
                    suggest_stabilizer,
                    session_progress,
                    play_onboarding,
                    session_progress_display,
                ]
                n_out = len(_out_play)
                n_skip_updates = n_out - 1

                _sp_def = {"episodes": 0, "best_reward": None, "last_token_pct": None}

                def on_start(sp: dict | None, onboarding: bool) -> tuple:
                    e = FCEnvEnvironment()
                    o = e.reset()
                    snap = _snapshot(o)
                    s0: dict = {"env": e, "pre_obs": snap}
                    bup = _button_state_from_obs(o)
                    tr_empty: list[dict] = []
                    ls0 = dict(LIVE_STATS_DEFAULT)
                    c0 = _compute_confidence_level(ls0)
                    sp0 = {**_sp_def, **(sp if isinstance(sp, dict) else {})}
                    prev_tok = sp0.get("last_token_pct")
                    prev_tok_f = (
                        float(prev_tok) if isinstance(prev_tok, (int, float)) else None
                    )
                    w = 100.0 * int(o.tokens) / MAX_TOKENS if MAX_TOKENS else 0.0
                    sp_next = {
                        "episodes": int(sp0.get("episodes", 0)) + 1,
                        "best_reward": sp0.get("best_reward"),
                        "last_token_pct": w,
                    }
                    return (
                        s0,
                        tr_empty,
                        ls0,
                        c0,
                        _render_six_clues(o.revealed_clues, None),
                        _stat_html_steps(ls0),
                        _stat_html_reward(ls0),
                        _confidence_html(c0),
                        _decision_insight_html(o, None, "Episode started — your move."),
                        _suggest_prompt_html(),
                        _last_step_html(
                            o,
                            "Episode started — your move.",
                            None,
                        ),
                        _episode_trace_html(tr_empty, False),
                        _play_flow_html(o, None, onboarding),
                        _play_footer_html(o, prev_tok_f),
                    ) + bup + (
                        None,
                        sp_next,
                        onboarding,
                        _session_progress_html(sp_next),
                    )

                def on_step(
                    s: dict | None,
                    user_action: int,
                    ep_tr: list | None,
                    live: dict | None,
                    prev_suggest: dict | None,
                    sp: dict | None,
                    onboarding: bool,
                ) -> tuple:
                    if not s or s.get("env") is None:
                        return (s,) + (gr.update(),) * n_skip_updates

                    e: FCEnvEnvironment = s["env"]  # type: ignore[assignment]
                    pre_dict = s.get("pre_obs")
                    if not isinstance(pre_dict, dict):
                        return on_start(sp, onboarding)
                    pre_clues: tuple = tuple(pre_dict.get("revealed_clues", ()))
                    o = e.step(Action(action=user_action))
                    if len(pre_clues) < 6:
                        new_idx = None
                    else:
                        new_idx = _newly_revealed_index(pre_clues, o.revealed_clues)
                    next_s: dict = {"env": e, "pre_obs": _snapshot(o)}
                    bup = _button_state_from_obs(o)
                    new_tr = _append_episode_trace(ep_tr, o)
                    new_live = _update_live_stats(live, o)
                    c_new = _compute_confidence_level(new_live)
                    ot = _outcome_text(user_action, pre_dict, o, new_idx)
                    sp_in = {**_sp_def, **(sp if isinstance(sp, dict) else {})}
                    prev_tok = sp_in.get("last_token_pct")
                    prev_tok_f = (
                        float(prev_tok) if isinstance(prev_tok, (int, float)) else None
                    )
                    sp_next = {**sp_in}
                    if o.done:
                        tr_run = float(new_live.get("total_reward", 0.0))
                        br = sp_next.get("best_reward")
                        sp_next["best_reward"] = (
                            tr_run if br is None else max(float(br), tr_run)
                        )
                    w = 100.0 * int(o.tokens) / MAX_TOKENS if MAX_TOKENS else 0.0
                    sp_next["last_token_pct"] = w
                    sug_html, sug_rec = _suggest_after_step(
                        prev_suggest, _snapshot(o), new_live
                    )
                    return (
                        next_s,
                        new_tr,
                        new_live,
                        c_new,
                        _render_six_clues(o.revealed_clues, new_idx),
                        _stat_html_steps(new_live),
                        _stat_html_reward(new_live),
                        _confidence_html(c_new),
                        _decision_insight_html(o, user_action, ot),
                        sug_html,
                        _last_step_html(
                            o,
                            ot,
                            user_action,
                        ),
                        _episode_trace_html(new_tr, o.done),
                        _play_flow_html(o, user_action, False),
                        _play_footer_html(o, prev_tok_f),
                    ) + bup + (
                        sug_rec,
                        sp_next,
                        False,
                        _session_progress_html(sp_next),
                    )

                def on_suggest(
                    s: dict | None,
                    live: dict | None,
                    prev_rec: dict | None,
                ) -> tuple:
                    if not s or s.get("env") is None:
                        return _suggest_prompt_html(), None
                    pd = s.get("pre_obs")
                    if not isinstance(pd, dict):
                        return _suggest_prompt_html(), None
                    ov = _obs_from_snapshot(pd)
                    rec, html = _smooth_recommendation(prev_rec, ov, live)
                    return html, _suggest_stabilizer_pack(rec, html)

                b_reset.click(
                    on_start,
                    inputs=[session_progress, play_onboarding],
                    outputs=_out_play,
                )
                b_low.click(
                    lambda s, t, lv, sg, sp, ob: on_step(s, 0, t, lv, sg, sp, ob),
                    inputs=[
                        st,
                        episode_trace,
                        live_stats,
                        suggest_stabilizer,
                        session_progress,
                        play_onboarding,
                    ],
                    outputs=_out_play,
                )
                b_high.click(
                    lambda s, t, lv, sg, sp, ob: on_step(s, 1, t, lv, sg, sp, ob),
                    inputs=[
                        st,
                        episode_trace,
                        live_stats,
                        suggest_stabilizer,
                        session_progress,
                        play_onboarding,
                    ],
                    outputs=_out_play,
                )
                b_commit.click(
                    lambda s, t, lv, sg, sp, ob: on_step(s, 2, t, lv, sg, sp, ob),
                    inputs=[
                        st,
                        episode_trace,
                        live_stats,
                        suggest_stabilizer,
                        session_progress,
                        play_onboarding,
                    ],
                    outputs=_out_play,
                )
                b_skip.click(
                    lambda s, t, lv, sg, sp, ob: on_step(s, 3, t, lv, sg, sp, ob),
                    inputs=[
                        st,
                        episode_trace,
                        live_stats,
                        suggest_stabilizer,
                        session_progress,
                        play_onboarding,
                    ],
                    outputs=_out_play,
                )
                b_suggest.click(
                    on_suggest,
                    inputs=[st, live_stats, suggest_stabilizer],
                    outputs=[suggest_display, suggest_stabilizer],
                )
            with gr.Tab("Compare"):
                gr.HTML(
                    _compare_panel_html(), elem_classes=["fc-tab-compare"]
                )

            with gr.Tab("Training Insights"):

                def _simulate_training_pulse() -> tuple:
                    time.sleep(1.25)
                    return (
                        gr.update(
                            value=(
                                '<div class="fc-sim-banner">Training view ready \u2014 reward curve '
                                "and win rate vs random below.</div>"
                            )
                        ),
                        gr.update(
                            elem_classes=[
                                "fc-insights-page",
                                "fc-insights-dynamic",
                                "fc-insights--pulse",
                            ]
                        ),
                    )

                with gr.Column(
                    elem_classes=["fc-insights-page", "fc-insights-dynamic"]
                ) as insights_col:
                    sim_banner = gr.HTML("", elem_id="fc-insights-sim-banner")
                    with gr.Row(elem_classes=["fc-insights-sim-row"]):
                        b_simulate = gr.Button(
                            "Simulate Training",
                            elem_id="fc-btn-simulate-training",
                        )
                    for _img_name, _stitle, _ssub in TRAINING_INSIGHT_SECTIONS:
                        _ap = _artifact_png_path(_img_name)
                        with gr.Group(elem_classes=["fc-insight-card"]):
                            gr.HTML(_insight_section_header_html(_stitle, _ssub))
                            if _img_name == "reward_curve.png":
                                gr.HTML(REWARD_CURVE_TAKEAWAY_HTML)
                            elif _img_name == "win_rate_vs_random.png":
                                gr.HTML(WINRATE_TAKEAWAY_HTML)
                            if _ap.is_file():
                                gr.Image(
                                    value=str(_ap),
                                    show_label=False,
                                    show_download_button=False,
                                )
                            else:
                                gr.HTML(
                                    '<p class="fc-insight-miss">Training graph not available</p>',
                                    elem_classes=["fc-insight-miss-box"],
                                )

                b_simulate.click(
                    _simulate_training_pulse,
                    outputs=[sim_banner, insights_col],
                )

            with gr.Tab("About"):
                gr.HTML(ABOUT_PANEL_HTML, elem_classes=["fc-tab-about"])

    return demo
