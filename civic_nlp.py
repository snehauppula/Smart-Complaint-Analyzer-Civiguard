"""Shared NLP logic: same preprocessing + priority rules as index.ipynb; loads saved NB pipeline."""


from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_KEEP = frozenset({
    "no", "not", "nor", "but", "since", "day", "days", "week", "weeks",
    "urgent", "severe", "severely", "over",
})
_FILLERS = frozenset({
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
})

_HIGH_WORDS = frozenset({
    "danger", "risk", "risks", "accident", "accidents", "fire", "fires",
})
_LONG_SINCE_DAYS_MIN = 3
_MED_WORDS = frozenset({
    "overflow", "overflowing", "leak", "leakage", "leaking", "frequent",
})


def _high_duration_triggers(low: str) -> list[tuple[str, str]]:
    """Return (machine_signal, user_facing_bullet) for timeline rules → HIGH."""
    found: list[tuple[str, str]] = []

    def add(signal: str, bullet: str) -> None:
        found.append((signal, bullet))

    m = re.search(r"since\s+(?:the\s+|last\s+|past\s+)?(\d+)\s+days?\b", low)
    if m and int(m.group(1)) >= _LONG_SINCE_DAYS_MIN:
        d = m.group(1)
        add(
            f"HIGH: long duration (since {d} days ≥ {_LONG_SINCE_DAYS_MIN})",
            f"Long timeline: 'since {d} days' meets our threshold ({_LONG_SINCE_DAYS_MIN}+ days) for escalation.",
        )

    for rx in (
        r"(?:for|going\s+on\s+for)\s+(?:the\s+)?(?:past\s+)?(\d+)\s+days?\b",
        r"(?:for|over)\s+the\s+last\s+(\d+)\s+days?\b",
    ):
        m2 = re.search(rx, low)
        if m2 and int(m2.group(1)) >= _LONG_SINCE_DAYS_MIN:
            d = m2.group(1)
            add(
                f"HIGH: long duration ({d} days)",
                f"Long timeline: the issue is described across {d} days — counted as prolonged.",
            )

    if re.search(r"\b(?:for\s+)?over\s+a\s+week\b", low) or re.search(
        r"\bmore\s+than\s+a\s+week\b", low
    ):
        add(
            "HIGH: duration (over a week)",
            "Long timeline: phrases like 'over a week' or 'more than a week' imply an extended unresolved period.",
        )

    if re.search(r"\b(?:for\s+|over\s+)?one\s+week\b", low) and not re.search(
        r"\bin\s+one\s+week\b", low
    ):
        add(
            "HIGH: duration (one week)",
            "Long timeline: one week (or more) of duration was mentioned.",
        )

    if re.search(r"\b(?:for\s+the\s+)?past\s+week\b", low):
        add(
            "HIGH: duration (past week)",
            "Long timeline: 'past week' is treated as roughly a week-long ongoing issue.",
        )

    m3 = re.search(r"\b(\d+)\s+weeks?\b", low)
    if m3 and int(m3.group(1)) >= 1:
        w = m3.group(1)
        add(
            f"HIGH: duration ({w} week(s))",
            f"Long timeline: {w} week(s) indicates an extended timeframe.",
        )

    return found

_lemmatizer = WordNetLemmatizer()
STOPWORDS: set[str] | None = None


def ensure_nltk() -> None:
    """Idempotent: safe after Streamlit hot-reload (globals reset; re-build STOPWORDS once)."""
    global STOPWORDS
    if STOPWORDS is not None:
        return
    for pkg in (
        "stopwords",
        "punkt_tab",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ):
        nltk.download(pkg, quiet=True)
    _en_stop = set(stopwords.words("english"))
    STOPWORDS = (_en_stop | _FILLERS) - _KEEP


def _penn_to_wordnet(tag: str):
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return wn.NOUN


def _remove_stopwords(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = text.strip().lower()
    if not text or text == "nan":
        return ""
    assert STOPWORDS is not None
    tokens = word_tokenize(text)
    kept = [w for w in tokens if w not in STOPWORDS]
    return " ".join(kept)


def _lemmatize_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = text.strip()
    if not text or text.lower() == "nan":
        return ""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens, lang="eng")
    lemmas = [_lemmatizer.lemmatize(w.lower(), _penn_to_wordnet(t)) for w, t in tagged]
    return " ".join(lemmas)


def preprocess_raw_complaint(raw: str) -> str:
    ensure_nltk()
    s = str(raw).strip().lower()
    s = re.sub(r"[^\w\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _remove_stopwords(s)
    s = _lemmatize_text(s)
    return s


def estimate_priority(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "level": "low",
            "score_1_10": 1,
            "raw_score": 0,
            "signals": ["empty text"],
            "explanation": {
                "headline": "No text to analyze",
                "bullets": ["Enter a complaint to run the rule-based priority scan."],
                "footer": None,
            },
        }

    low = str(text).lower()
    toks = set(re.findall(r"[a-z]{2,}", low))
    signals: list[str] = []
    bullets: list[str] = []
    high = False

    hit_w = _HIGH_WORDS & toks
    if hit_w:
        high = True
        kw = ", ".join(sorted(hit_w))
        signals.append("HIGH: " + kw)
        bullets.append(
            f"Safety / risk language: the words ({kw}) often indicate hazards or harm — escalated for fast review."
        )

    if "exposed wire" in low:
        high = True
        signals.append('HIGH: phrase "exposed wire"')
        bullets.append(
            'Electrical hazard: the phrase \"exposed wire\" maps to a high-priority public-safety pattern.'
        )

    for sig, bl in _high_duration_triggers(low):
        high = True
        signals.append(sig)
        bullets.append(bl)

    if high:
        score = min(10, 7 + len(signals))
        return {
            "level": "high",
            "score_1_10": score,
            "raw_score": 10,
            "signals": signals,
            "explanation": {
                "headline": "High priority — safety or long-running issue",
                "bullets": bullets,
                "footer": "Priority is rule-based (keywords + timelines), separate from the ML category.",
            },
        }

    med_signals: list[str] = []
    med_bullets: list[str] = []
    hit_m = _MED_WORDS & toks
    if hit_m:
        km = ", ".join(sorted(hit_m))
        med_signals.append("MEDIUM: " + km)
        med_bullets.append(
            f"Service impact: terms like ({km}) suggest spills, backups, or repeat problems worth faster routing."
        )
    if "frequent issue" in low:
        med_signals.append('MEDIUM: phrase "frequent issue"')
        med_bullets.append(
            "Recurrence: 'frequent issue' signals the problem keeps happening — medium escalation."
        )
    if re.search(r"\bfrequently\b", low) and "frequent issue" not in low:
        med_signals.append("MEDIUM: frequently")
        med_bullets.append(
            "Recurrence: 'frequently' suggests repeated incidents — medium escalation."
        )

    if med_signals:
        return {
            "level": "medium",
            "score_1_10": 5,
            "raw_score": 5,
            "signals": med_signals,
            "explanation": {
                "headline": "Medium priority — operational or recurring cues",
                "bullets": med_bullets,
                "footer": "Adjust keyword lists in code if your municipality uses different phrasing.",
            },
        }

    return {
        "level": "low",
        "score_1_10": 2,
        "raw_score": 1,
        "signals": ["LOW: general complaint (no HIGH/MEDIUM rules matched)"],
        "explanation": {
            "headline": "Standard priority — no escalation rules matched",
            "bullets": [
                "No high-tier triggers: safety words (fire, danger, accident, risk), 'exposed wire', or long timelines we detect.",
                f"No long timeline: we look for {_LONG_SINCE_DAYS_MIN}+ days (e.g. 'since 4 days'), week-scale phrases ('over a week', 'past week'), or explicit weeks.",
                "No medium-tier triggers: overflow, leakage, or frequent / recurring wording.",
            ],
            "footer": "Tip: add concrete duration ('since 5 days', 'over a week') or hazard detail if the issue is urgent.",
        },
    }


def resolve_model_path() -> Path:
    name = "complaint_nb_pipeline.joblib"
    here = Path(__file__).resolve().parent
    candidates = [here / name, Path.cwd() / name]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Missing {name}. Train the model in index.ipynb (joblib.dump) or copy the file next to the app."
    )


def load_pipeline() -> Any:
    return joblib.load(resolve_model_path())


def analyze_complaint(raw: str, pipeline: Any) -> dict[str, Any]:
    cleaned = preprocess_raw_complaint(raw)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = list(pipeline.classes_)
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])
    pri = estimate_priority(raw)
    return {
        "category": str(pred),
        "confidence": confidence,
        "cleaned_preview": cleaned[:200] + ("…" if len(cleaned) > 200 else ""),
        "priority": pri,
        "all_proba": {str(c): float(p) for c, p in zip(classes, proba)},
    }
