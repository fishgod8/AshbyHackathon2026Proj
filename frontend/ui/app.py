"""Pathology Assistant: Gradio workbench UI.

Framing (see CLAUDE.md, PRODUCT.md):
  A. AI pre-read / triage   Queue + Review: every tile is pre-read by the AI. Confident tiles are
                            cleared automatically; only flagged tiles enter the review queue, and
                            each decision (accept / correct) advances to the next one.
  B. Pathology dictionary   Lookup: upload an ambiguous tile, compare it side by side with the
                            closest confirmed reference tiles of each candidate class.
  C. Evidence               When the "confident" label can be trusted.

Data: <data folder>/cases.json (+ optional meta.json, evidence.json, ihc.json, examples/).
With --live, the Lookup tab analyzes new tiles through the same integration layer (src/live.py).

Run: python app.py                         # mock data (noise placeholders)
     python app.py --data data_semi        # real tiles + stub models (build with: python -m src.analyze)
     python app.py --data data_semi --live # + Lookup tab (loads QuiltNet, a few seconds)
then open http://127.0.0.1:7860
"""
import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
# Repo layout: ui/app.py + ../src. Deployed bundle (deploy/build_space.py): workbench.py next to src/.
REPO = HERE if (HERE / "src").is_dir() else HERE.parent

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="mock_data", help="folder under ui/ holding cases.json")
parser.add_argument(
    "--live", nargs="?", default=None, const=str(REPO / "embeddings/quiltnet_train_subset.csv"),
    help="enable the Lookup tab; optional gallery embedding CSV",
)
parser.add_argument(
    "--session-decisions", action="store_true",
    help="keep sign-offs per browser session only (public demo); default also saves them to decisions.json",
)
parser.add_argument("--share", action="store_true", help="also create a temporary public *.gradio.live link")
parser.add_argument("--embedded", action="store_true",
                    help="rendered inside ui/site.py (dataset switch): the site draws the page header")
parser.add_argument("--show-review", action="store_true",
                    help="show the Queue and Review tabs (hidden for now; Lookup and Evidence are the demo)")
args, _ = parser.parse_known_args()

DATA_DIR = Path(__file__).parent / args.data
CASES = json.loads((DATA_DIR / "cases.json").read_text())
CASE_BY_ID = {c["case_id"]: c for c in CASES}


def load_optional(name):
    path = DATA_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


META = load_optional("meta.json") or {"source": "mock data"}
CLASSES = META.get("classes") or sorted(CASES[0]["probabilities"])
CLASS_NAMES = META.get("class_names", {})
REF_NOUN = META.get("reference_noun", "tiles")  # "tiles" or "lesions", depending on what labels mean
UNIT, UNITS = ("image", "images") if REF_NOUN == "images" else ("tile", "tiles")  # what one case is
SIMILARITY_NOTE = (f'<p class="small muted sim-note">{escape(META["similarity_note"])}</p>'
                   if META.get("similarity_note") else "")
# Evidence numbers (see docs/INTERFACE.md); the tab shows a notice if missing.
EVIDENCE = load_optional("evidence.json")
# Optional IHC knowledge table: {"disclaimer": ..., "rules": [{"classes": [...], "stains": [...], "why": ..., "source": ...}]}
IHC = load_optional("ihc.json")
NEXT_STEPS = load_optional("next_steps.json")  # what to do next per candidate pair (tests first, then general steps)
# Optional PubMed papers per differential pair (src/literature.py); only reviewed ones are shown up front.
LITERATURE = load_optional("literature.json")

# Reviewer decisions: {case_id: {"action": "accept" | "correct", "label": ..., "time": ...}}.
# They live in each browser session's state; locally they are also saved to decisions.json.
DECISIONS_PATH = None if args.session_decisions else DATA_DIR / "decisions.json"

# Live analysis for Lookup: same integration layer as the precomputed cases.
LIVE = None
LIVE_CACHE = Path(tempfile.gettempdir()) / "ashby_live" / DATA_DIR.name  # copies of retrieved gallery images
if args.live:
    sys.path.insert(0, str(REPO))
    from src.analyze import localize_result

    if META.get("pipeline") == "bach":
        # BACH: Mayumi's 12-patch QuiltNet embedding + her classifier + Ashley's flag (src/bach.py).
        from src.bach import BachLiveAnalyzer

        LIVE = BachLiveAnalyzer(DATA_DIR)
    elif META.get("pipeline") == "breakhis":
        # BreakHis: Mayumi's 6-patch embeddings + a temporary classifier + Ashley's checks (src/breakhis.py).
        from src.breakhis import BreakHisLiveAnalyzer

        LIVE = BreakHisLiveAnalyzer(DATA_DIR)
    else:
        from src.live import LiveAnalyzer

        LIVE = LiveAnalyzer(Path(args.live))
    (LIVE_CACHE / "images").mkdir(parents=True, exist_ok=True)

# Slide view (BACH): slides precomputed by src/slides.py (shown only, never used for training); photos live.
SLIDES, SLIDE_LIB, SLIDE_PHOTOS = {}, None, []
if LIVE is not None and (DATA_DIR / "patches" / "index.csv").exists():
    import numpy as np
    from PIL import Image as PILImage

    from src import slides as slide_tools
    from src.patches import PatchLibrary

    SLIDE_LIB = PatchLibrary(DATA_DIR)
    SLIDE_CLF = slide_tools.patch_classifier(DATA_DIR)  # Mayumi's single-patch model; Lookup keeps her photo model
    for _m in sorted((DATA_DIR / "slides").glob("*/meta.json")):
        _meta = json.loads(_m.read_text())
        SLIDES[_meta["name"]] = _meta
    SLIDE_PHOTOS = sorted(p for p in (DATA_DIR / "examples").glob("*") if p.suffix in (".png", ".jpg", ".webp"))
SLIDE_VIEW = SLIDE_LIB is not None and bool(SLIDES or SLIDE_PHOTOS)
_sv_val = DATA_DIR / "slides" / "validation.json"
SLIDE_VALIDATION = json.loads(_sv_val.read_text()) if SLIDE_VIEW and _sv_val.exists() else []
SLIDE_VALIDATION_NOTE = (
    "Annotated training slides from the same challenge, compared patch by patch with the regions pathologists drew; "
    "never trained on. Settings were chosen on the first three and checked unchanged on the others. AUC: how often a "
    "cancer patch scores above a non-cancer one (0.5 is chance, 1 is perfect).")


def slide_validation_sentence(rows):
    """What the unseen slides showed, from the numbers: weakest slide, and where the photo-level model did better."""
    unseen = [r for r in rows if r["role"] == "Unseen check"]
    if not unseen:
        return ""
    worst = min(unseen, key=lambda r: r["auc"])
    text = f"On the unseen slides the map was weaker ({worst['slide']}: {worst['auc']:.2f})"
    better = [r for r in unseen if r.get("auc_photo_model", 0) > r["auc"]]
    if better:
        text += (", and the earlier whole-photo model scored higher there (" +
                 ", ".join(f"{r['slide']} {r['auc_photo_model']:.2f}" for r in better) + ")")
    return text + ": a first attempt at where to look, not a detector."

# Risk value from the uncertainty module -> (label, css tone, sort order). Unknown values sort with "review".
RISK = {
    "high": ("High risk", "bad", 0),
    "review": ("Needs review", "warn", 1),
    "confident": ("Cleared automatically", "ok", 2),
}
N_COMPARE = 3  # candidate classes shown side by side
VIEWS = ["Needs review", "Cleared automatically", "All"]


def risk_info(risk):
    return RISK.get(risk, (risk, "warn", 1))


def full_name(code):
    name = CLASS_NAMES.get(code)
    return f"{name} ({code})" if name and name != code else code


def top_prob(c):
    return max(c["probabilities"].values())


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


# ---- Queue order and decisions ----

# Most urgent first: by risk level, then least confident first.
ORDERED = sorted(CASES, key=lambda c: (risk_info(c["risk"])[2], top_prob(c)))
FLAGGED = [c for c in ORDERED if c["risk"] != "confident"]
CLEARED = [c for c in ORDERED if c["risk"] == "confident"]
QUEUE_POS = {c["case_id"]: i for i, c in enumerate(FLAGGED, start=1)}


def load_decisions():
    if DECISIONS_PATH is None or not DECISIONS_PATH.exists():
        return {}
    return json.loads(DECISIONS_PATH.read_text())


def save_decisions(decided):
    if DECISIONS_PATH is not None:
        DECISIONS_PATH.write_text(json.dumps(decided, indent=2))


def pending(decided):
    return [c for c in FLAGGED if c["case_id"] not in decided]


def next_pending(decided, after=None):
    """Next undecided flagged case after `after` in queue order, wrapping to the start."""
    todo = [c["case_id"] for c in pending(decided) if c["case_id"] != after]
    if not todo:
        return None
    if after in QUEUE_POS:
        later = [cid for cid in todo if QUEUE_POS[cid] > QUEUE_POS[after]]
        if later:
            return later[0]
    return todo[0]


def decision_label(d):
    if not d:
        return ""
    return "Accepted" if d["action"] == "accept" else f"Corrected to {d['label']}"


# ---- HTML pieces ----

def dot(tone):
    return f'<span class="dot dot-{tone}" aria-hidden="true"></span>'


def app_header_html():
    return (
        '<header class="app-header"><div>'
        '<h1 class="app-title">Pathology Assistant</h1>'
        '<p class="app-subtitle">AI pre-read and reference lookup for H&amp;E histology</p>'
        f'</div><span class="source-pill">{escape(META["source"])}</span></header>'
    )


def queue_head_html(decided):
    n, left = len(FLAGGED), len(pending(decided))
    done = n - left
    return (
        '<div class="queue-head"><div>'
        '<h2 class="page-title">Review queue</h2>'
        f'<p class="lede">{n} tiles need a pathologist. {len(CLEARED)} were cleared automatically '
        "because the AI was confident.</p></div>"
        f'<div class="progress-wrap"><div class="progress-label"><b>{done}</b> of {n} reviewed</div>'
        f'<progress max="{max(n, 1)}" value="{done}" aria-label="Review progress"></progress></div></div>'
    )


def start_button(decided):
    left, n = len(pending(decided)), len(FLAGGED)
    if left == 0:
        return gr.Button("Queue complete", interactive=False, variant="primary")
    label = "Start review" if left == n else f"Continue review ({left} left)"
    return gr.Button(label, interactive=True, variant="primary")


def cleared_note_html():
    if EVIDENCE and EVIDENCE.get("error_rate_confident") is not None:
        rate = pct(EVIDENCE["error_rate_confident"])
        return (f'<p class="lede">Not sent for review. In the evidence set, {rate} of tiles the AI marked '
                "confident were wrong, so occasional spot-checks are worthwhile.</p>")
    return '<p class="lede">Not sent for review because the AI was confident.</p>'


def queue_table(view, decided):
    source = {"Needs review": FLAGGED, "Cleared automatically": CLEARED}.get(view, ORDERED)
    rows = []
    for c in source:
        label, tone, _ = risk_info(c["risk"])
        d = decided.get(c["case_id"])
        if d:
            decision = escape(decision_label(d))
        elif c["risk"] == "confident":
            decision = '<span class="muted">Not needed</span>'
        else:
            decision = '<span class="muted">Pending</span>'
        rows.append([
            f'<span class="status">{dot(tone)}{escape(label)}</span>',
            c["case_id"],
            full_name(c["prediction"]),
            round(100 * top_prob(c)),
            ", ".join(x for x in c["candidate_set"] if x != c["prediction"]),
            decision,
        ])
    return pd.DataFrame(rows, columns=["Status", "Case", "Suggested", "Confidence %", "Could also be", "Decision"])


def case_header_html(c, decided):
    label, tone, _ = risk_info(c["risk"])
    cid = c["case_id"]
    position = f"{QUEUE_POS[cid]} of {len(FLAGGED)} in queue" if cid in QUEUE_POS else "Not in the review queue"
    d = decided.get(cid)
    decision = decision_label(d) or ("Awaiting decision" if cid in QUEUE_POS else "No decision needed")
    return (
        f'<div class="case-head"><h2 class="page-title">{escape(cid)}</h2>'
        f'<span class="pill pill-{tone}">{dot(tone)}{escape(label)}</span>'
        f'<span class="meta">{escape(position)}</span><span class="meta" aria-hidden="true">·</span>'
        f'<span class="meta">{escape(decision)}</span></div>'
    )


def ihc_rules_html(pair):
    """Curated stain rules (ihc.json) for this pair; empty when there is no table or no matching rule."""
    if not IHC:
        return ""
    hits = [r for r in IHC["rules"] if set(r["classes"]) == set(pair)]
    if not hits:
        return ""
    items = "".join(
        f'<li><b>{escape(", ".join(r["stains"]))}</b> <span class="muted">{escape(r.get("why", ""))} '
        f'(source: {escape(r.get("source", "none given"))})</span></li>'
        for r in hits
    )
    return f'<ul class="rules">{items}</ul><p class="small muted">{escape(IHC.get("disclaimer", "For reference only."))}</p>'


def papers_html(entry, show_screened=3):
    """Three tiers: reviewed by a pathologist, AI-screened with a verified quote, then the rest folded."""
    if not entry or not entry["papers"]:
        return '<p class="small muted">PubMed returned no papers for this pair.</p>'

    def item(p, badge):
        quote = (p.get("ai") or {}).get("quote") or ""
        quote_html = f'<p class="quote">\u201c{escape(quote)}\u201d</p>' if quote else ""
        return (f'<li>{badge}<a href="{escape(p["url"])}" target="_blank" rel="noopener">{escape(p["title"])}</a>'
                f'<div class="paper-meta">{escape(p["first_author"])} et al. · {escape(p["journal"])} · {escape(p["year"])}</div>'
                f'{quote_html}</li>')

    papers = entry["papers"]
    reviewed = [p for p in papers if p.get("reviewed")]
    screened = [p for p in papers if not p.get("reviewed") and (p.get("ai") or {}).get("verified")]
    shown_ids = {p["pmid"] for p in reviewed + screened[:show_screened]}
    rest = [p for p in papers if p["pmid"] not in shown_ids]

    out = ""
    if reviewed:
        badge = '<span class="pill pill-ok">Reviewed by a pathologist</span> '
        out += f'<ul class="papers">{"".join(item(p, badge) for p in reviewed)}</ul>'
    if screened:
        badge = ""  # no per-paper label; the note under the panel title says the papers were AI-screened
        out += f'<ul class="papers">{"".join(item(p, badge) for p in screened[:show_screened])}</ul>'
    if not reviewed and not screened:
        out += ('<p class="small muted">AI screening found no paper that clearly addresses this differential.</p>'
                if entry.get("screening") else '<p class="small muted">No papers reviewed by a pathologist yet.</p>')
    if rest:
        out += (f'<details><summary>{len(rest)} other search results</summary>'
                f'<ul class="papers">{"".join(item(p, "") for p in rest)}</ul></details>')
    relaxed = " (context words dropped: the full search found nothing)" if entry.get("relaxed") else ""
    if entry.get("url"):
        out += (f'<a class="search-link" href="{escape(entry["url"])}" target="_blank" rel="noopener">'
                f'Open this search on PubMed ({entry["count"]} results)</a><span class="small muted">{relaxed}</span>')
    return out


# Header note for the references panel: say plainly what the AI did and did not do.
REFERENCES_SOURCE = (
    "Papers from PubMed, screened by AI. Quotes are verbatim, checked against each abstract."
    if LITERATURE and LITERATURE.get("_meta", {}).get("ai_screening")
    else "Papers from PubMed. No AI-written text."
)


def next_steps_html(c):
    """Suggested next step for a flagged differential: tests for this pair, then general steps. Empty if none."""
    alts = [x for x in c["candidate_set"] if x != c["prediction"]]
    if not NEXT_STEPS or not alts:
        return ""
    pair = {c["prediction"], alts[0]}
    entry = next((e for e in NEXT_STEPS["pairs"] if set(e["classes"]) == pair), None)
    items = []
    for t in (entry or {}).get("tests", []):
        srcs = " · ".join(f'<a href="{escape(s["url"])}" target="_blank" rel="noopener">{escape(s["citation"])}</a>'
                          for s in t["sources"])
        quotes = "".join(f'<p class="quote">\u201c{escape(s["quote"])}\u201d <span class="muted">({escape(s["citation"])})</span></p>'
                         for s in t["sources"])
        name = f'Stain: {t["name"]}' if t.get("kind", "stain") == "stain" else t["name"]
        items.append(f'<li><b>{escape(name)}</b><div class="step-read">{escape(t["read"])}</div>'
                     f'<details class="step-src"><summary>Source: {srcs}</summary>{quotes}</details></li>')
    if entry is not None and not entry.get("tests"):
        items.append('<li><span class="muted">No test listed yet for this pair; to be added with a pathologist.</span></li>')
    items += [f'<li><b>{escape(g["action"])}</b></li>' for g in NEXT_STEPS.get("general", [])]
    a, b = full_name(c["prediction"]), full_name(alts[0])
    return (f'<div class="next-steps"><div class="next-title">Suggested next step for {escape(a)} vs {escape(b)}</div>'
            f'<ol>{"".join(items)}</ol><p class="small muted step-status">{escape(NEXT_STEPS.get("status_note", ""))}</p></div>')


def references_html(c):
    """Literature and stains for the top two candidates. Empty when the case has no differential."""
    alts = [x for x in c["candidate_set"] if x != c["prediction"]]
    if not alts or not (LITERATURE or IHC):
        return ""
    pair = sorted([c["prediction"], alts[0]])
    lit = (LITERATURE or {}).get("|".join(pair))
    if lit is None and not ihc_rules_html(pair):
        return ""  # pair never searched (e.g. background is not a tissue): nothing honest to show
    lit = lit or {}
    a, b = full_name(c["prediction"]), full_name(alts[0])
    body = (
        f'<p class="small muted ref-source">{REFERENCES_SOURCE}</p>'
        '<div class="references-grid">'
        f'<section><div class="ref-title">Telling them apart</div>{papers_html(lit.get("differential"))}</section>'
        f'<section><div class="ref-title">Stains that may help</div>{ihc_rules_html(pair)}{papers_html(lit.get("ihc"))}</section>'
        '</div>'
    )
    if NEXT_STEPS:  # the next step leads; papers are background, folded (pathologist feedback, 10/18)
        return (f'<details class="references refs-fold"><summary class="subhead">Background papers for {escape(a)} '
                f'vs {escape(b)}</summary>{body}</details>')
    return f'<div class="references"><h3 class="subhead">References for {escape(a)} vs {escape(b)}</h3>{body}</div>'


def references_update(c):
    html = references_html(c)
    return gr.HTML(value=html, visible=bool(html))


def verdict_html(c, mode="case", show_truth=False):
    """The hero block: suggestion, alternatives, neighbor check, next step."""
    probs, pred = c["probabilities"], c["prediction"]
    alts = [x for x in c["candidate_set"] if x != pred]
    agree, n = sum(s["label"] == pred for s in c["similar_cases"]), len(c["similar_cases"])
    support = c.get("support")
    # mode "patch": one patch of a slide or photo (Slide view); its references are patches, not whole images
    unit, units, noun = ("patch", "patches", "patches") if mode == "patch" else (UNIT, UNITS, REF_NOUN)
    label = {"case": "Suggested", "lookup": "Closest match", "patch": "This patch"}.get(mode, "Closest match")
    if is_unfamiliar(c) and mode in ("lookup", "patch"):  # no class, no comparisons: only the warning and what to do
        todo = ("Look at this patch directly; the AI's reading is not reliable here." if mode == "patch" else
                "Try the original photo: uncompressed, at full scanner resolution.")
        return (
            f'<div class="verdict"><div class="label">{label}</div>'
            f'<div class="verdict-main">Unfamiliar {unit}</div>'
            f'<div class="note note-warn">It is farther from the training {units} than {support["percentile"]}% of '
            f"validation {units} were, so the AI gives no class for it. Compression, another magnification or "
            f'another tissue type can cause this.</div><div class="next">{todo}</div></div>'
        )
    if is_unfamiliar(c):  # review queue: the class is not the headline when the input is outside what the model knows
        parts = [
            f'<div class="label">{"Suggested" if mode == "case" else "Closest match"}</div>',
            f'<div class="verdict-main">Unfamiliar {UNIT}</div>',
            f'<div class="note note-warn">It is farther from the training {UNITS} than {support["percentile"]}% of '
            f"validation {UNITS} were, so the class probabilities below are less reliable. Compression, another magnification or "
            "another tissue type can cause this.</div>",
        ]
    else:
        parts = [
            f'<div class="label">{label}</div>',
            f'<div class="verdict-main">{escape(full_name(pred))}</div>',
            f'<div class="verdict-conf">{probs[pred]:.0%} confidence</div>',
        ]
    if alts:
        alt_txt = ", ".join(f"<b>{escape(full_name(a))}</b> {probs.get(a, 0):.0%}" for a in alts)
        parts.append(f'<div class="verdict-alt"><span class="muted">Could also be</span> {alt_txt}</div>')
    if n == 5 and agree <= LOW_AGREE:  # evidence, not a new review rule: the similar images point elsewhere
        then = (f" In the test set, the AI was right only {LOW_AGREE_RIGHT:.0%} of the time when this happened."
                if LOW_AGREE_RIGHT is not None and mode != "patch" else "")  # that rate is per image, not per patch
        which = (f"None of the {n} most similar confirmed {noun} are" if agree == 0 else
                 f"Only {agree} of the {n} most similar confirmed {noun} {'is' if agree == 1 else 'are'}")
        parts.append(f'<div class="note note-warn"><b>Mixed evidence.</b> {which} {escape(full_name(pred))}.{then}</div>')
    steps = next_steps_html(c) if c["risk"] != "confident" and mode != "patch" else ""  # Slide view stays light
    if steps:  # flagged differential: what to do next comes before comparing pictures
        parts.append(steps)
    elif mode == "patch":
        parts.append('<div class="next">The most similar confirmed patches are below. A pointer, not a diagnosis.</div>')
    elif mode == "lookup":
        parts.append('<div class="next">Compare the candidates below. Reference lookup, not a diagnosis.</div>')
    elif c["risk"] == "confident":
        parts.append('<div class="next">Cleared automatically. You are spot-checking it.</div>')
    else:
        parts.append('<div class="next">Compare the candidates below, then accept or correct.</div>')
    if show_truth and c.get("true_label"):
        truth, flagged = c["true_label"], c["risk"] != "confident"
        if truth == pred:
            outcome = "The AI suggestion was correct."
        elif flagged:
            outcome = "The AI suggestion was wrong, but it was flagged for review."
        else:
            outcome = "The AI suggestion was wrong and it was not flagged."
        parts.append(f'<div class="truth">Demo only. Ground truth: <b>{escape(full_name(truth))}</b>. {escape(outcome)}</div>')
    return '<div class="verdict">' + "".join(parts) + "</div>"


# ---- Galleries ----

def to_gallery(refs, base, caption):
    return [(str(base / s["image"]), caption(s)) for s in refs]


def compare_updates(c, base):
    """One gallery per candidate class (suggested first), closest confirmed references in each."""
    cands = c.get("candidate_examples")
    if not cands:  # older data without candidate_examples
        cands = {c["prediction"]: [s for s in c["similar_cases"] if s["label"] == c["prediction"]][:3]}
        if c.get("contrast_cases"):
            cands[c["contrast_cases"][0]["label"]] = c["contrast_cases"]
    items = list(cands.items())[:N_COMPARE]
    updates = []
    for i in range(N_COMPARE):
        if i < len(items):
            label, refs = items[i]
            role = "suggested" if label == c["prediction"] else "alternative"
            updates.append(gr.Gallery(
                value=to_gallery(refs, base, lambda s: f"{s['detail']} {s['score']:.2f}" if s.get("detail")
                                 else f"similarity {s['score']:.2f}"),
                label=f"{full_name(label)}, {role}", visible=True,
            ))
        else:
            updates.append(gr.Gallery(value=None, visible=False))
    return updates


def prob_bars_html(c, top_n=5):
    """Flat bars for the top classes; candidates in the accent color, the rest neutral."""
    items = sorted(c["probabilities"].items(), key=lambda kv: -kv[1])[:top_n]
    rows = "".join(
        f'<div class="prob-row"><div class="prob-head"><span>{escape(full_name(code))}</span>'
        f'<span class="prob-val">{p:.0%}</span></div><div class="prob-track">'
        f'<div class="prob-fill {"bar-strong" if code in c["candidate_set"] else "bar-soft"}" '
        f'style="width:{100 * p:.1f}%"></div></div></div>'
        for code, p in items
    )
    return f'<div class="probs"><div class="label">Class probabilities</div>{rows}</div>'


def is_unfamiliar(c):
    """Reference-support check (Ashley): the image is unlike what the model learned from."""
    return bool(c.get("support") and c["support"]["flagged"])


def why_html(c, mode="case"):
    """Shown under the decision: the uncertainty module's reason and the top class probabilities."""
    if is_unfamiliar(c) and mode in ("lookup", "patch"):  # distance only; probabilities would suggest a class
        return (f'<div class="why"><div class="why-title">Why it was flagged</div>'
                f'<p class="why-reason">{escape(c["support"]["reason"])}</p></div>')
    if mode == "patch":
        title = "Why this reading"
    elif mode == "lookup":
        title = "Why this match"
    elif c["risk"] == "confident":
        title = "Why it was cleared"
    else:
        title = "Why it was flagged"
    pred = c["prediction"]
    agree, n = sum(s["label"] == pred for s in c["similar_cases"]), len(c["similar_cases"])
    bars = "" if mode in ("lookup", "patch") else prob_bars_html(c, top_n=4)  # shown in the result column there
    noun = "patches" if mode == "patch" else REF_NOUN
    patients = {s["group"] for s in c["similar_cases"] if s.get("group")}
    from_patients = ("" if not patients else " They all come from one patient." if len(patients) == 1
                     else f" They come from {len(patients)} different patients.")
    return (f'<div class="why"><div class="why-title">{title}</div>'
            f'<p class="why-reason">{escape(c["reason"])} {agree} of the {n} most similar confirmed {noun} '
            f'{"is" if agree == 1 else "are"} {escape(pred)}.{from_patients}</p>{bars}</div>')


def why_outputs(c, base, mode="case"):
    """(inline why block, nearest-neighbor gallery for the 'more similar' accordion)."""
    neighbors = to_gallery(c["similar_cases"], base, lambda s: f"{s.get('detail') or s['label']} {s['score']:.2f}")
    return why_html(c, mode), neighbors


# ---- Event handlers ----

def queue_outputs(view, decided):
    return queue_table(view, decided), queue_head_html(decided), start_button(decided)


def change_view(view, decided):
    cleared = view == "Cleared automatically"
    return queue_table(view, decided), gr.HTML(cleared_note_html(), visible=cleared), gr.Button(visible=cleared)


def open_case(evt: gr.SelectData):
    """Clicked a queue row -> open that case."""
    return evt.row_value[1], gr.Tabs(selected="case")


def start_review(decided):
    nxt = next_pending(decided)
    return (nxt if nxt else gr.skip()), gr.Tabs(selected="case")


def spot_check():
    if not CLEARED:
        gr.Warning("No cleared tiles to spot-check.")
        return gr.skip(), gr.skip()
    return random.choice(CLEARED)["case_id"], gr.Tabs(selected="case")


def show_case(case_id, show_truth, decided):
    c = CASE_BY_ID[case_id]
    return (
        case_header_html(c, decided),
        str(DATA_DIR / c["image"]),
        verdict_html(c, "case", show_truth),
        *compare_updates(c, DATA_DIR),
        references_update(c),
        *why_outputs(c, DATA_DIR),
        # The most likely correction is the runner-up candidate: preselect it, keep the panel closed.
        gr.Dropdown(value=next((x for x in c["candidate_set"] if x != c["prediction"]), None)),
        gr.Row(visible=False),
    )


def decide(case_id, action, label, view, decided):
    """Record a decision in this session (and decisions.json locally), then advance to the next flagged tile."""
    decided = {**decided, case_id: {"action": action, "label": label,
                                    "time": datetime.now().isoformat(timespec="seconds")}}
    save_decisions(decided)
    nxt = next_pending(decided, after=case_id)
    if nxt is None:
        gr.Info("Queue complete. Every flagged tile has a decision.")
    return (nxt or case_id), case_header_html(CASE_BY_ID[case_id], decided), *queue_outputs(view, decided), decided


def accept(case_id, view, decided):
    return decide(case_id, "accept", CASE_BY_ID[case_id]["prediction"], view, decided)


def correct(case_id, new_label, view, decided):
    if not new_label:
        gr.Warning("Choose the correct class first.")
        return (gr.skip(),) * 6
    return decide(case_id, "correct", new_label, view, decided)


def skip(case_id, decided):
    nxt = next_pending(decided, after=case_id)
    if nxt is None:
        gr.Info("No other tiles are waiting for review.")
        return gr.skip()
    return nxt


LOOKUP_EMPTY = (
    f'<div class="empty"><div class="empty-title">No {UNIT} yet</div>'
    f'<p class="lede">Pick an example above or upload your own {UNIT}, then choose Look up. '
    f"You will see the closest match, why, and confirmed reference {UNITS} for each candidate.</p></div>"
)


# ---- Slide view ----

SV_LAYERS = {"Suspicious": "abnormal", "AI unsure": "unsure", "Experts": "annotation"}
SV_PHOTO_WIDTH = 1200
_SV_CACHE = {}
EXPERT = {0: "not annotated (other tissue)", 1: "Benign", 2: "In situ carcinoma", 3: "Invasive carcinoma"}


def sv_choices():
    return ([f"Slide {name}" for name in SLIDES] +
            [f"Photo {i} (BACH test)" for i in range(1, len(SLIDE_PHOTOS) + 1)])


def sv_source(choice):
    """(grid, base image, scale, stride, reader, annotation overlay or None) for a slide or photo; cached."""
    if choice in _SV_CACHE:
        return _SV_CACHE[choice]
    if choice.startswith("Slide "):
        name = choice.removeprefix("Slide ")
        meta, folder = SLIDES[name], DATA_DIR / "slides" / name
        z = np.load(folder / "grid.npz")
        grid = {k: z[k] for k in z.files} | {"tile": meta["tile"], "meta": meta}
        grid["rank"] = slide_tools.suspicion_rank(grid)  # scanned slides: relative, not absolute (see Evidence)
        base = PILImage.open(folder / "overview.jpg").convert("RGB")
        annotation = PILImage.open(folder / "annotation.png") if (folder / "annotation.png").exists() else None

        def reader(i, folder=folder, meta=meta, grid=grid):
            import openslide

            with openslide.OpenSlide(str(folder / "slide.svs")) as sl:
                return sl.read_region((int(grid["x"][i]), int(grid["y"][i])), 0, (meta["tile"],) * 2).convert("RGB")
        entry = (grid, base, base.width / meta["width"], None, reader, annotation)
    else:
        img = PILImage.open(SLIDE_PHOTOS[int(choice.split(" ")[1]) - 1]).convert("RGB")
        entry = sv_photo_entry(img)
    _SV_CACHE[choice] = entry
    return entry


def sv_photo_entry(img):
    grid = slide_tools.analyze_photo(img, LIVE.embedder, SLIDE_CLF, SLIDE_LIB, LIVE.threshold)
    base = img.resize((SV_PHOTO_WIDTH, round(img.height * SV_PHOTO_WIDTH / img.width)))
    reader = lambda i, img=img, grid=grid: img.crop((int(grid["x"][i]), int(grid["y"][i]),  # noqa: E731
                                                    int(grid["x"][i]) + 224, int(grid["y"][i]) + 224))
    return grid, base, SV_PHOTO_WIDTH / img.width, 112, reader, None


def sv_layers(entry):
    """Slides: suspicious + expert annotation (the classifier's own uncertainty means little on scans, see the
    summary line). Photos: suspicious + AI unsure, as on the rest of the site."""
    if "meta" in entry[0]:
        return ["Suspicious"] + (["Experts"] if entry[5] is not None else [])
    return ["Suspicious", "AI unsure"]


def swatch(rgb, text):
    return f'<span><i style="background:rgb{rgb}"></i>{escape(text)}</span>'


def sv_summary_html(entry, layer):
    """One count line, a legend for the layer on screen, and one plain sentence about how far to trust it."""
    grid = entry[0]
    n, slide = len(grid["x"]), "meta" in grid
    if layer == "Experts":
        legend = "".join(swatch(slide_tools.ANNOTATION_RGB[k], CLASS_NAMES.get(k, k)) for k in ("Benign", "InSitu", "Invasive"))
    elif layer == "AI unsure":
        legend = swatch(slide_tools.LOW_CONF_RGB, f"Low confidence, {np.mean(grid['low_conf']):.0%} of patches")
        if not slide:
            legend += swatch(slide_tools.UNFAMILIAR_RGB, f"Unfamiliar tissue, {np.mean(grid['unfamiliar']):.0%}")
    elif slide:
        legend = swatch(slide_tools.ABNORMAL_RGB, "Most suspicious 20% of this slide (darker = more suspicious)")
    else:
        legend = swatch(slide_tools.ABNORMAL_RGB, "Likely in situ or invasive carcinoma (50% or more)")
    if slide and grid["meta"].get("validation"):
        v = grid["meta"]["validation"]
        trust = (f"Experts marked cancer in {v['cancer_share']:.0%} of this slide's tissue and in "
                 f"{v['top20_cancer_share']:.0%} of its most suspicious 20%. A guide to where to look first. "
                 "Slides are never used for training.")
    elif slide:
        trust = "A guide to where to look first. Slides are never used for training."
    else:
        trust = "Colors suggest where to look first; they are not a diagnosis."
    what = f"{n:,} tissue patches" if slide else f"{n} overlapping patches"
    return (f'<div class="sv-summary"><div class="sv-count">{what}</div><div class="hund-legend sv-legend">{legend}'
            f'</div><p class="small muted sv-trust">{trust}</p></div>')


SV_EMPTY = ('<div class="empty sv-empty"><div class="empty-title">Click a spot to inspect it</div><p class="lede">'
            "You will see that patch enlarged, how suspicious it is, and the most similar confirmed patches.</p></div>")


def sv_render(entry, layer, sel=None):
    grid, base, scale, stride, _, annotation = entry
    return slide_tools.render(base, grid, SV_LAYERS[layer], scale, stride, highlight=sel, annotation=annotation)


def sv_open(choice, layer, entry):
    layers = sv_layers(entry)
    layer = layer if layer in layers else layers[0]
    return (sv_render(entry, layer), sv_summary_html(entry, layer), {"choice": choice, "sel": None},
            gr.Image(visible=False), SV_EMPTY, gr.HTML(visible=False), gr.Gallery(value=None, visible=False),
            gr.Radio(choices=layers, value=layer))


def sv_show(choice, layer):
    return sv_open(choice, layer, sv_source(choice))


def sv_upload(file, layer, state):
    if file is None:
        return (gr.skip(),) * 9
    img = PILImage.open(file).convert("RGB")
    if min(img.size) < 224:
        gr.Warning("The photo must be at least 224 × 224 pixels.")
        return (gr.skip(),) * 9
    choice = f"Your photo ({img.width}×{img.height})"
    _SV_CACHE[choice] = sv_photo_entry(img)
    # The source menu names what is on screen, so it does not keep saying the slide shown before
    return (*sv_open(choice, layer, _SV_CACHE[choice]), gr.Dropdown(choices=[*sv_choices(), choice], value=choice))


def sv_relayer(layer, state):
    entry = sv_source(state["choice"])
    return sv_render(entry, layer, state.get("sel")), sv_summary_html(entry, layer)


def sv_patch_case(grid, i):
    """One patch in the result format the Lookup blocks render."""
    classes = [str(c) for c in grid["classes"]]
    p = grid["probs"][i].astype(float)
    order = np.argsort(p)[::-1]
    pred, runner = classes[order[0]], classes[order[1]]
    slide = "meta" in grid
    low, unf = bool(grid["low_conf"][i]), bool(grid["unfamiliar"][i]) and not slide  # slides: said once, not per patch
    emb = grid["emb"][i].astype(np.float32)
    support = {"flagged": unf, "distance": round(float(grid["distance"][i]), 4), "threshold": SLIDE_LIB.gate.config["threshold"],
               "k": SLIDE_LIB.gate.config["k"], "percentile": SLIDE_LIB.gate.config["percentile"]}
    support["reason"] = (f"The 5th closest training patch is at cosine distance {support['distance']:.3f} "
                         f"(review limit {support['threshold']:.3f}).")
    conf_reason = (f"{pred} ({p[order[0]]:.0%}) is below the review threshold of {LIVE.threshold:.0%}; runner-up "
                   f"{runner} ({p[order[1]]:.0%})." if low else
                   f"{pred} scores {p[order[0]]:.0%}, at or above the review threshold of {LIVE.threshold:.0%}.")
    return {"prediction": pred, "probabilities": {classes[j]: round(float(p[j]), 3) for j in order},
            "candidate_set": [pred, runner] if low else [pred], "risk": "review" if (low or unf) else "confident",
            "reason": f"{conf_reason} {support['reason']}", "support": support,
            "similar_cases": SLIDE_LIB.similar(emb, k=5)}


def sv_slide_verdict(grid, i, case):
    """Slides: where the patch ranks within its slide, what its closest confirmed patches are and what the experts
    drew there. The classifier's own confidence comes from microscope photos and does not carry over to scans."""
    rank = grid["rank"][i]
    hot = rank >= 1 - slide_tools.TOP_SHARE
    sims = Counter(s["label"] for s in case["similar_cases"]).most_common()
    parts = ['<div class="label">This patch</div>',
             f'<div class="verdict-main">{"Suspicious area" if hot else "Not flagged"}</div>',
             f'<div class="verdict-conf">More suspicious than {rank:.0%} of this slide\'s tissue</div>',
             '<p class="sv-fact"><span class="muted">Most similar confirmed patches</span> '
             + ", ".join(f"{n} {escape(CLASS_NAMES.get(lab, lab))}" for lab, n in sims) + "</p>"]
    if "truth" in grid:
        parts.append(f'<p class="sv-fact"><span class="muted">Expert annotation here</span> '
                     f'{escape(EXPERT[int(grid["truth"][i])])}</p>')
    return f'<div class="verdict">{"".join(parts)}</div>'


def sv_click(evt: gr.SelectData, layer, state):
    entry = sv_source(state["choice"])
    grid, base, scale, stride, reader, _ = entry
    i = slide_tools.patch_at(grid, evt.index[0] / scale, evt.index[1] / scale)
    if i is None:
        gr.Info("No tissue patch here. Click on the tissue.")
        return (gr.skip(),) * 6
    case = sv_patch_case(grid, i)
    if "meta" in grid:
        verdict = sv_slide_verdict(grid, i, case)
        p = case["probabilities"][case["prediction"]]
        inner = (f'<p class="small">The classifier alone reads {escape(full_name(case["prediction"]))} ({p:.0%}). '
                 "It learned from microscope photos, and on scanned slides its confidence is not reliable, so the "
                 "map ranks patches within the slide instead.</p>" + prob_bars_html(case, top_n=4))
    else:
        verdict = verdict_html(case, "patch")
        inner = why_html(case, "patch") + ("" if is_unfamiliar(case) else prob_bars_html(case, top_n=4))
    details = f'<details class="refs-fold sv-details"><summary>Details</summary>{inner}</details>'
    similar = [(s["image"], s["label"]) for s in case["similar_cases"]]  # class only: five thumbnails in a narrow column
    return (sv_render(entry, layer, i), {**state, "sel": i}, gr.Image(value=reader(i).resize((280, 280)), visible=True),
            verdict, gr.HTML(value=details, visible=True), gr.Gallery(value=similar, visible=True))


def lookup(image):
    """Uploaded tile -> live analysis -> same rendering as a queued case."""
    if image is None:
        gr.Warning(f"Upload {'an image' if UNIT == 'image' else 'a tile'} or pick an example first.")
        empty = [gr.Gallery(value=None, visible=False)] * N_COMPARE
        return (LOOKUP_EMPTY, gr.HTML(visible=False), gr.HTML(visible=False), *empty, gr.HTML(visible=False),
                gr.Accordion(visible=False), [])
    try:
        analyzed = LIVE.analyze(image)
    except ValueError as e:  # e.g. the BACH model needs at least 224 x 224 pixels
        gr.Warning(str(e))
        empty = [gr.Gallery(value=None, visible=False)] * N_COMPARE
        return (LOOKUP_EMPTY, gr.HTML(visible=False), gr.HTML(visible=False), *empty, gr.HTML(visible=False),
                gr.Accordion(visible=False), [])
    result = localize_result(analyzed, LIVE_CACHE / "images")
    why, neighbors = why_outputs(result, LIVE_CACHE, "lookup")
    if is_unfamiliar(result):  # e.g. a beach photo must not be compared with "the closest in situ carcinoma"
        hidden = [gr.Gallery(value=None, visible=False)] * N_COMPARE
        return (verdict_html(result, "lookup"), gr.HTML(value=why, visible=True), gr.HTML(visible=False), *hidden,
                gr.HTML(visible=False), gr.Accordion(visible=False), [])
    verdict = verdict_html(result, "lookup") + f'<div class="lk-probs">{prob_bars_html(result, top_n=4)}</div>'
    return (
        verdict, gr.HTML(value=why, visible=True), gr.HTML(visible=True),
        *compare_updates(result, LIVE_CACHE), references_update(result), gr.Accordion(visible=True), neighbors,
    )


def model_table(models):
    return pd.DataFrame([
        {
            "Setting": m["model"],
            "Right overall": pct(m["accuracy"]),
            "Sent to a pathologist": pct(m["review_rate"], 0),
            "Wrong among the rest": pct(m["error_rate_confident"]),
        }
        for m in models
    ])


def by_patient_table(rows):
    return pd.DataFrame([{"Patient": r["patient"], "Diagnosis": f'{r["truth"]}, {r["subtype"].lower()}',
                          f"{UNITS.capitalize()}": r["n"], "Wrong": f'{r["wrong"]} ({r["wrong"] / r["n"]:.0%})',
                          "Mean confidence": f'{r["mean_confidence"]:.0%}', "Sent for review": r["review"]}
                         for r in rows])


def robustness_table(rb):
    n = rb["n"]
    return pd.DataFrame([{"Input": r["input"], "Suggestion changed": f"{r['changed']} of {n}",
                          "Flagged as unfamiliar": f"{r['unfamiliar']} of {n}",
                          "Sent for review": f"{r['review']} of {n}"} for r in rb["rows"]])


def slide_validation_table(rows):
    role = {"Chosen on": "Used to choose settings", "Unseen check": "Unseen"}
    return pd.DataFrame([{"Slide": r["slide"], "Role": role.get(r["role"], r["role"]), "Cancer in its tissue": f"{r['cancer_share']:.0%}",
                          "Cancer in its most suspicious 20%": f"{r['top20_cancer_share']:.0%}",
                          "AUC": f"{r['auc']:.2f}"} for r in rows])


def robustness_sentence(rb):
    """The row where the suggestion moved the most, said plainly."""
    worst = max(rb["rows"][1:], key=lambda r: r["changed"], default=None)
    if not worst or not worst["changed"]:
        return ""
    return (f"{worst['input']}: the AI changed its suggestion for {worst['changed']} of {rb['n']} {UNITS}, and the "
            f"unfamiliar-{UNIT} check flagged {worst['unfamiliar']} of {rb['n']}.")


LOW_AGREE = 3  # warn when at most this many of the 5 most similar confirmed images share the AI's answer


def agreement(c):
    """How many of the most similar confirmed images have the AI's suggested class."""
    return sum(s["label"] == c["prediction"] for s in c["similar_cases"])


LABELED = [c for c in CASES if c.get("true_label")]
_low = [c for c in LABELED if len(c["similar_cases"]) == 5 and agreement(c) <= LOW_AGREE]
# Share the AI got right in the test set when the similar images disagreed (quoted in the Lookup warning)
LOW_AGREE_RIGHT = (sum(c["prediction"] == c["true_label"] for c in _low) / len(_low)) if len(_low) >= 5 else None


def ev_bar(key, value, strong=True):
    return (f'<div class="ev-line"><span class="ev-key">{key}</span><div class="prob-track"><div class="prob-fill '
            f'{"bar-strong" if strong else "bar-soft"}" style="width:{100 * value:.1f}%"></div></div>'
            f'<span class="ev-val">{value:.0%}</span></div>')


def ev_row(label, n, bars):
    small = " · small sample" if n < 20 else ""  # a handful of images: don't read much into the bar
    return (f'<div class="ev-row"><div class="ev-label">{escape(label)}<span class="ev-n">{n:,} {UNITS}{small}</span></div>'
            f'<div class="ev-bars-col">{bars}</div></div>')


def per_hundred_html(cases):
    """Who handles what, out of every 100 images: the AI alone vs the AI plus a pathologist for the unsure ones."""
    n = len(cases)
    wrong = [c["prediction"] != c["true_label"] for c in cases]
    sure = [c["risk"] == "confident" for c in cases]
    handled = round(100 * sum(sure) / n)
    review = 100 - handled
    all_wrong = round(100 * sum(wrong) / n)
    alone_wrong = round(100 * sum(w and k for w, k in zip(wrong, sure)) / n)
    caught = all_wrong - alone_wrong  # rounded once, so the parts add up

    def bar(segments):
        return '<div class="hund-bar">' + "".join(
            f'<span class="seg {cls}" style="flex:{v} 0 0"></span>' for cls, v in segments if v > 0) + "</div>"

    caught_txt = ("all of them" if alone_wrong == 0 and caught else f"{caught} of them" if caught else "none of them")
    take = (f'<p class="ev-take">Out of every 100 {UNITS}, letting the AI handle all of them gives {all_wrong} mistakes. '
            f"Letting it handle only the {handled} it is sure about, and sending {review} to a pathologist, puts "
            f"{caught_txt} in front of a person; {alone_wrong or 'none'} stay in the {UNITS} the AI handled alone.</p>")
    rows = (
        f'<div class="hund-row"><div class="ev-label">AI handles everything</div>'
        f'{bar([("seg-ok", 100 - all_wrong), ("seg-bad", all_wrong)])}'
        f'<div class="hund-txt">100 by the AI, <b>{all_wrong} wrong</b></div></div>'
        f'<div class="hund-row"><div class="ev-label">AI handles only what it is sure about</div>'
        f'{bar([("seg-ok", handled - alone_wrong), ("seg-bad", alone_wrong), ("seg-rev-bad", caught), ("seg-rev", review - caught)])}'
        f'<div class="hund-txt">{handled} by the AI, <b>{alone_wrong} wrong</b><br>{review} to a pathologist, '
        f"{caught} of them AI mistakes</div></div>"
    )
    legend = ('<div class="hund-legend"><span><i class="seg-ok"></i>AI right</span><span><i class="seg-bad"></i>AI wrong'
              '</span><span><i class="seg-rev-bad"></i>AI wrong, sent to a pathologist</span><span><i class="seg-rev"></i>'
              "AI right, sent to a pathologist</span></div>")
    note = (f'<p class="small muted hund-note">Each bar is 100 {UNITS}, scaled from the {n:,} test {UNITS}. Whether the '
            "pathologist then gets them right is not measured here.</p>")
    return take + '<div class="hund">' + rows + legend + "</div>" + note


def calibration_html(cases):
    """Does '90% sure' mean 90% right? Claimed vs actual, in three confidence bands."""
    bands = [("AI was at least 90% sure", 0.9, 1.01), ("AI was 70 to 90% sure", 0.7, 0.9), ("AI was under 70% sure", 0, 0.7)]
    rows, top = [], None
    for label, lo, hi in bands:
        cs = [c for c in cases if lo <= max(c["probabilities"].values()) < hi]
        if not cs:
            continue
        claimed = sum(max(c["probabilities"].values()) for c in cs) / len(cs)
        right = sum(c["prediction"] == c["true_label"] for c in cs) / len(cs)
        top = top or (claimed, right, len(cs))
        rows.append(ev_row(label, len(cs), ev_bar("Claimed", claimed, strong=False) + ev_bar("Right", right)))
    claimed, right, n = top
    gap = round(100 * (claimed - right))
    verdict = ("so here its confidence can be taken at face value" if gap <= 3
               else f"so it is about {gap} points more sure of itself than it should be")
    return (f'<p class="ev-take">When the AI said it was at least 90% sure, it was right {right:.0%} of the time '
            f"(it claimed {claimed:.0%} on average), {verdict}.</p>" + '<div class="ev-bars">' + "".join(rows) + "</div>")


def agreement_html(cases):
    """Is the AI more often wrong when the most similar confirmed images disagree with it?"""
    bands = [("All 5 similar images agree", 5, 5), ("4 of 5 agree", 4, 4), (f"{LOW_AGREE} or fewer agree", 0, LOW_AGREE)]
    rows, rate = [], {}
    for label, lo, hi in bands:
        cs = [c for c in cases if len(c["similar_cases"]) == 5 and lo <= agreement(c) <= hi]
        if not cs:
            continue
        rate[lo] = sum(c["prediction"] == c["true_label"] for c in cs) / len(cs)
        rows.append(ev_row(label, len(cs), ev_bar("Right", rate[lo])))
    take = ""
    if 5 in rate and 0 in rate:
        take = (f'<p class="ev-take">When all 5 of the most similar confirmed {UNITS} had the AI\'s answer, it was right '
                f"{rate[5]:.0%} of the time; when {LOW_AGREE} or fewer did, {rate[0]:.0%}. Lookup warns in that case.</p>")
    return take + '<div class="ev-bars">' + "".join(rows) + "</div>"


def evidence_head_html(e):
    verb = "was" if e["n_wrong_confident"] == 1 else "were"
    target = f" (target at least {pct(1 - e['alpha'], 0)})" if e.get("alpha") else ""
    return (
        '<h2 class="page-title">Can “confident” be trusted?</h2>'
        f'<p class="headline">Of the {e["n_confident"]:,} {UNITS} the AI marked confident, '
        f'{e["n_wrong_confident"]:,} {verb} wrong ({pct(e["error_rate_confident"])}).</p>'
        f'<p class="lede">{pct(e["review_rate"], 0)} of {UNITS} were sent to a pathologist; the AI handled the other '
        f'{pct(1 - e["review_rate"], 0)} on its own{target}.</p>'
        f'<p class="source">{escape(e["source"])}</p>'
    )


APP_FOOTER = (
    '<footer class="app-footer">Tiles: NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al., '
    '<a href="https://zenodo.org/records/1214456" target="_blank" rel="noopener">Zenodo 1214456</a>, CC BY 4.0). '
    "Embeddings: QuiltNet-B-32. Literature: PubMed (NCBI). "
    "Research demo with placeholder models; not for clinical use.</footer>"
)


# ---- Look & feel ----

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    # Tables and labels use the "mono" font in Gradio; keep the whole workbench in one typeface.
    font_mono=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f6f7f9",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    button_primary_background_fill="#1d4f91",
    button_primary_background_fill_hover="#163f75",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#1d4f91",
)

CSS = """
:root {
  --bg: oklch(0.975 0.004 255); --surface: oklch(1 0 0);
  --ink: oklch(0.23 0.02 260); --ink-2: oklch(0.38 0.02 260); --muted: oklch(0.50 0.02 260);
  --line: oklch(0.91 0.008 255); --hover: oklch(0.965 0.012 255);
  --accent: oklch(0.42 0.12 255);
  --ok: oklch(0.45 0.11 150); --ok-dot: oklch(0.64 0.15 150); --ok-bg: oklch(0.96 0.03 150);
  --warn: oklch(0.47 0.10 65); --warn-dot: oklch(0.77 0.15 75); --warn-bg: oklch(0.965 0.04 85);
  --bad: oklch(0.48 0.16 27); --bad-dot: oklch(0.62 0.19 27); --bad-bg: oklch(0.96 0.03 27);
  --radius: 12px; --ease: cubic-bezier(0.25, 1, 0.5, 1);
  /* Spacing scale (4 pt): tight inside a group, generous between sections */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px; --s-5: 24px; --s-6: 32px; --s-7: 48px;
}
.gradio-container { width: 100% !important; max-width: 1180px !important; margin: 0 auto !important;
                    overflow: visible !important; overflow-x: clip !important; }  /* clip, not hidden: keeps sticky working */

/* Header */
.app-header { display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
              padding:8px 2px 16px; border-bottom:1px solid var(--line); }
.app-title { font-size:22px; font-weight:700; color:var(--ink); letter-spacing:-0.01em; margin:0; }
.app-subtitle { font-size:14px; color:var(--muted); margin:2px 0 0; }
.source-pill { font-size:12px; color:var(--ink-2); border:1px solid var(--line); border-radius:999px;
               padding:4px 10px; background:var(--surface); white-space:nowrap; }

/* Type */
.page-title { font-size:20px; font-weight:650; color:var(--ink); letter-spacing:-0.01em; margin:0; text-wrap:balance; }
.lede { font-size:14px; line-height:1.5; color:var(--ink-2); margin:4px 0 0; max-width:72ch; }
.headline { font-size:22px; line-height:1.3; font-weight:600; color:var(--ink); margin:10px 0 0; max-width:60ch; text-wrap:balance; }
.source { font-size:12px; color:var(--muted); margin:8px 0 0; }
.label { font-size:13px; font-weight:500; color:var(--muted); }
.muted { color:var(--muted); }
.meta { font-size:13px; color:var(--muted); }
.placeholder { font-size:15px; color:var(--muted); padding:24px 4px; }
.section-title h3 { font-size:15px !important; font-weight:600 !important; color:var(--ink) !important; margin:var(--s-6) 0 0 !important; }

/* Status */
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex:none; }
.dot-ok { background:var(--ok-dot); } .dot-warn { background:var(--warn-dot); } .dot-bad { background:var(--bad-dot); }
.status { display:inline-flex; align-items:center; gap:8px; }
.pill { display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px; }
.pill-ok { color:var(--ok); background:var(--ok-bg); }
.pill-warn { color:var(--warn); background:var(--warn-bg); }
.pill-bad { color:var(--bad); background:var(--bad-bg); }

/* Queue */
.queue-head { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap; }
.progress-wrap { min-width:240px; flex:0 1 280px; }
.progress-label { font-size:13px; color:var(--ink-2); margin-bottom:6px; }
.progress-wrap progress { -webkit-appearance:none; appearance:none; display:block; width:100%; height:6px;
                          border:0; border-radius:999px; background:var(--line); overflow:hidden; }
.progress-wrap progress::-webkit-progress-bar { background:var(--line); border-radius:999px; }
.progress-wrap progress::-webkit-progress-value { background:var(--accent); border-radius:999px; }
.progress-wrap progress::-moz-progress-bar { background:var(--accent); border-radius:999px; }
#queue-table .virtual-row { cursor:pointer; }
#queue-table .virtual-row:hover .body-cell { background:var(--hover) !important; }

/* Review */
.case-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.verdict-main { font-size:28px; font-weight:650; color:var(--ink); line-height:1.2; letter-spacing:-0.01em; margin-top:2px; text-wrap:balance; }
.verdict-conf { font-size:15px; color:var(--ink-2); margin-top:2px; }
.verdict-alt { font-size:15px; color:var(--ink-2); margin-top:var(--s-3); }
.note { font-size:14px; color:var(--ink-2); background:oklch(0.955 0.006 255); border-radius:8px; padding:var(--s-2) var(--s-3); margin-top:var(--s-4); }
.note ul { margin:6px 0 0 18px; }
.note-warn { color:var(--warn); background:var(--warn-bg); }
.next { font-size:14px; font-weight:600; color:var(--accent); margin-top:var(--s-4); }
/* Suggested next step: tests for the differential first, then general steps */
.next-steps { margin-top:var(--s-5); border:1px solid var(--line); border-radius:var(--radius); padding:var(--s-4); background:var(--surface); }
.next-title { font-size:15px; font-weight:600; color:var(--ink); }
.next-steps ol { margin:var(--s-3) 0 0 var(--s-5); padding:0; display:flex; flex-direction:column; gap:var(--s-3); }
.next-steps li { font-size:14px; color:var(--ink); line-height:1.4; padding-left:var(--s-1); }
.step-read { font-size:13px; line-height:1.45; color:var(--ink-2); margin-top:var(--s-1); max-width:62ch; }
.step-src summary { cursor:pointer; font-size:12px; color:var(--muted); margin-top:var(--s-1); }
.step-src summary a { color:var(--ink-2); }
.step-src .quote { font-size:13px; line-height:1.45; margin:var(--s-2) 0 0; padding:var(--s-2) var(--s-3); border-radius:8px; background:var(--hover); color:var(--ink-2); }
.step-status { margin:var(--s-4) 0 0; padding-top:var(--s-3); border-top:1px solid var(--line); }
/* Folded sections share one look: the papers fold matches the gallery accordion below it */
.refs-fold { margin-top:var(--s-6); border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); padding:0 var(--s-3); }
.refs-fold > summary { list-style:none; cursor:pointer; display:flex; justify-content:space-between; align-items:center;
    gap:var(--s-3); margin:0 !important; padding:10px 0; font-size:14px; font-weight:500; color:var(--ink); }
.refs-fold > summary::-webkit-details-marker { display:none; }
.refs-fold > summary::after { content:"▼"; font-size:14px; line-height:1; color:var(--ink-2); transform:rotate(90deg);
    transition:transform 150ms var(--ease); }  /* same glyph, size and turn as Gradio's accordion icon */
.refs-fold[open] > summary::after { transform:none; }
.gradio-container .block:has(> .label-wrap) { border-color:var(--line) !important; }  /* same edge as the papers fold */
.gradio-container .label-wrap { font-size:14px !important; font-weight:500 !important; color:var(--ink) !important; }
.gradio-container .label-wrap .icon { font-size:14px !important; color:var(--ink-2) !important; }
.refs-fold[open] { padding-bottom:var(--s-3); }
.truth { font-size:13px; color:var(--ink-2); border:1px dashed oklch(0.75 0.01 260); border-radius:8px; padding:8px 12px; margin-top:12px; }

/* Images: the tissue is the hero; no box, rounded corners, scaled up from 224 px */
#case-image, #lookup-image { background:transparent !important; border:none !important; box-shadow:none !important; }
#case-image .image-frame { width:min(100%, 440px) !important; height:auto !important; }
/* Whole image, aspect kept (square CRC tiles, 4:3 BACH photos) */
#case-image .image-frame img { width:100% !important; height:auto !important;
    aspect-ratio:auto; object-fit:contain; border-radius:var(--radius); }
#lookup-image img { width:min(100%, 440px) !important; max-width:100%; height:auto !important;
    aspect-ratio:auto; object-fit:contain; border-radius:var(--radius); }
#case-image .image-container button { justify-content:flex-start; }

/* Compact controls */
.toolbar { align-items:center !important; }
#truth-toggle { background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important; }
#truth-toggle label { font-size:13px; color:var(--ink-2); justify-content:flex-end; }
.cleared-row { align-items:center !important; }
.flush, .flush .html-container { padding:0 !important; }
.compare .grid-wrap { padding:0 !important; }
.compare [data-testid="block-label"] { margin-left:0 !important; gap:0 !important; }
.compare [data-testid="block-label"] span { margin:0 !important; padding:0 !important; }

/* Galleries: flat, no frame or toolbar; the candidate name reads as a small heading */
.compare.block { background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important; }
.compare [data-testid="block-label"] { position:static !important; border:none !important; background:transparent !important;
    box-shadow:none !important; padding:0 0 6px !important; font-size:13px !important; font-weight:600 !important;
    color:var(--ink-2) !important; }
.compare [data-testid="block-label"] svg, .compare [data-testid="block-label"] span:has(svg) { display:none !important; }
.compare .grid-container { padding-top:0 !important; }

/* Why: the reason and probabilities, right under the decision */
.why { margin-top:var(--s-5); padding-top:var(--s-4); border-top:1px solid var(--line); }
.why-title { font-size:13px; font-weight:600; color:var(--ink-2); }
.why-reason { font-size:14px; line-height:1.45; color:var(--ink-2); margin:var(--s-1) 0 var(--s-3); max-width:68ch; }
.ref-source { margin:2px 0 0; }
.app-footer { margin-top:24px; padding-top:12px; border-top:1px solid var(--line); font-size:12px; color:var(--muted); }
.app-footer a { color:var(--ink-2); padding:0 !important; }
#verdict, #lk-verdict { padding:0 !important; }
.actions { gap:10px !important; margin-top:6px; align-items:center !important; }
.actions button, .correction button { white-space:nowrap; }
.link-btn { background:transparent !important; border-color:transparent !important; box-shadow:none !important;
            color:var(--accent) !important; font-weight:500 !important; }
.link-btn:hover { background:var(--hover) !important; }
.correction { gap:10px !important; padding:10px; margin-top:2px; background:var(--surface);
              border:1px solid var(--line); border-radius:var(--radius); align-items:center !important; }
/* Class probabilities: flat bars, no gradients */
.probs { max-width:420px; }
.prob-row { display:grid; grid-template-columns:minmax(0,1fr) 96px; align-items:center; column-gap:12px; margin-top:6px; }
.prob-head { display:flex; justify-content:space-between; gap:8px; font-size:13px; color:var(--ink-2); }
.prob-val { font-variant-numeric:tabular-nums; color:var(--ink); font-weight:500; }
.prob-track { height:4px; background:var(--line); border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; }
.bar-strong { background:var(--accent); }
.bar-soft { background:oklch(0.74 0.025 255); }
.subhead { font-size:15px; font-weight:600; color:var(--ink); margin:var(--s-6) 0 0; }
.small { font-size:13px; }
.sim-note { margin:2px 0 0; }
.privacy { margin:var(--s-1) 0 0; }
.lk-probs { margin-top:var(--s-5); }
/* Evidence: plain-language bars (claimed vs right, agreement) */
.ev-take, .ev-md p { font-size:14px; line-height:1.5; color:var(--ink-2); margin:var(--s-2) 0 0; max-width:75ch; }
.ev-bars { margin-top:var(--s-3); max-width:760px; }
.ev-row { display:grid; grid-template-columns:240px minmax(0, 1fr); gap:var(--s-2) var(--s-5); align-items:center;
          padding:var(--s-3) 0; border-top:1px solid var(--line); }
.ev-row:last-child { border-bottom:1px solid var(--line); }
.ev-label { font-size:14px; color:var(--ink); font-weight:500; display:flex; flex-direction:column; gap:2px; }
.ev-n { font-size:12px; color:var(--muted); font-weight:400; font-variant-numeric:tabular-nums; }
.ev-bars-col { display:flex; flex-direction:column; gap:var(--s-2); }
.ev-line { display:grid; grid-template-columns:64px minmax(0, 1fr) 44px; align-items:center; gap:var(--s-3); }
.ev-key { font-size:13px; color:var(--ink-2); }
.ev-val { font-size:13px; color:var(--ink); font-weight:500; text-align:right; font-variant-numeric:tabular-nums; }
/* Per 100 images: one stacked bar per way of working; text carries the numbers */
.hund { margin-top:var(--s-3); max-width:900px; }
.hund-row { display:grid; grid-template-columns:240px minmax(0, 1fr) 290px; gap:var(--s-2) var(--s-5); align-items:center;
            padding:var(--s-3) 0; border-top:1px solid var(--line); }
.hund-bar { display:flex; gap:2px; height:16px; }
.hund-bar .seg:first-child { border-radius:4px 0 0 4px; } .hund-bar .seg:last-child { border-radius:0 4px 4px 0; }
.seg-ok { background:oklch(0.62 0.09 255); } .seg-bad { background:var(--bad-dot); } .seg-rev { background:oklch(0.86 0.012 255); }
.seg-rev-bad { background:oklch(0.62 0.19 27 / 0.38); }
.hund-txt { font-size:13px; line-height:1.45; color:var(--ink-2); font-variant-numeric:tabular-nums; }
.hund-txt b { color:var(--ink); font-weight:600; }
.hund-legend { display:flex; flex-wrap:wrap; gap:var(--s-2) var(--s-5); padding-top:var(--s-3); border-top:1px solid var(--line);
               font-size:12px; color:var(--ink-2); }
.hund-legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.hund-note { margin:var(--s-2) 0 0; }
@media (max-width: 640px) { .ev-row, .hund-row { grid-template-columns:1fr; } }

/* References: literature and stains for the differential */
.references { margin-top:4px; }
.references-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:8px 32px; margin-top:8px; }
.references section { border-top:1px solid var(--line); padding-top:10px; }
.ref-title { font-size:13px; font-weight:600; color:var(--ink-2); margin-bottom:4px; }
.papers, .rules { list-style:none; margin:4px 0 8px; padding:0; }
.papers li, .rules li { margin:0 0 10px; padding-left:0 !important; font-size:14px; line-height:1.4; }
.papers li::before, .rules li::before { content:none !important; }
.papers a { color:var(--accent); font-weight:500; text-decoration:none; padding:0 !important; }
.papers a:hover { text-decoration:underline; text-underline-offset:2px; }
.paper-meta { font-size:12px; color:var(--muted); margin-top:2px; }
.references details summary { cursor:pointer; font-size:13px; color:var(--ink-2); margin:2px 0 6px; }
.references .quote { display:block; margin:6px 0 0; padding:6px 10px; border-radius:8px; background:var(--hover);
                     font-size:13px; line-height:1.45; color:var(--ink-2); }
.search-link { display:inline-block; margin-top:4px; padding:0 !important; font-size:13px; color:var(--accent); font-weight:500; text-decoration:none; }
.search-link:hover { text-decoration:underline; text-underline-offset:2px; }
.empty { border:1px dashed oklch(0.82 0.01 255); border-radius:var(--radius); padding:28px 24px; background:var(--surface);
          min-height:340px; display:flex; flex-direction:column; justify-content:center; }
.empty-title { font-size:16px; font-weight:600; color:var(--ink); }
.thumbnail-item { border-radius:8px !important; overflow:hidden; }
/* Compare galleries: fixed thumbnail size, so a lone candidate doesn't stretch and crop its tiles */
.compare .grid-container { grid-template-columns:repeat(auto-fill, minmax(150px, 172px)) !important; }
.thumbnail-item img { border-radius:8px; }

/* Lookup: input column (examples, image, button) beside the result column, both starting at the top */
#lookup-image .image-frame, #lookup-image .image-container { height:auto !important; }
#lk-left { gap:var(--s-3) !important; }
.row:has(> #lk-left) { column-gap:var(--s-6) !important; }  /* clear gutter between input and result */
/* Six examples in one row, filling the input column */
#lk-left .gallery { display:grid !important; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:var(--s-2) !important; }
#lk-left .gallery-item { padding:0 !important; min-width:0; overflow:hidden; border-radius:8px !important; }
#lk-left .gallery-item .container { display:block !important; width:100% !important; max-width:none !important;
    height:auto !important; aspect-ratio:4 / 3; border:0 !important; padding:0 !important; }
#lk-left .gallery-item > span { display:block; }
#lk-left .gallery-item img { width:100% !important; height:100% !important; object-fit:cover; display:block; }
/* Gallery accordion spacing matches the papers fold above it */
.gradio-container .block.accordion { margin-top:var(--s-3); }
/* Tables read like the rest of the page: same font, no extra chrome */
.gradio-container table, .gradio-container table *,
.gradio-container .virtual-row, .gradio-container .virtual-row * { font-family:inherit !important; font-variant-numeric:tabular-nums; }
.gradio-container .virtual-row, .gradio-container .virtual-row * , .gradio-container table { color:var(--ink) !important; }
.gradio-container table th { white-space:normal !important; }
/* Narrow screens: header stacks instead of squeezing the source pill */
@media (max-width: 640px) {
  .app-header { flex-direction:column; align-items:flex-start; gap:var(--s-2); }
  .verdict-main { font-size:24px; }
}

/* Slide view */
.sv-controls { gap:var(--s-3) !important; align-items:center !important; margin-bottom:var(--s-3); flex-wrap:nowrap !important; }
.sv-controls > * { min-width:0 !important; }
/* Layer switch: same segmented control as the dataset switch */
#sv-layer { flex:0 0 auto !important; width:max-content !important; overflow:visible !important; }
#sv-layer .wrap:has(> label) { position:relative; display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:0;
    padding:3px; border:1px solid var(--line); border-radius:999px; background:var(--surface); }
#sv-layer .wrap:has(> label)::before { content:""; position:absolute; top:3px; bottom:3px; left:3px;
    width:calc((100% - 6px) / 2); border-radius:999px; background:var(--accent); transition:transform 280ms var(--ease); }
#sv-layer .wrap:has(> label:nth-child(2).selected)::before { transform:translateX(100%); }
#sv-layer .wrap:has(> label:nth-child(3))::before { width:calc((100% - 6px) / 3); }
#sv-layer .wrap:has(> label:nth-child(3).selected)::before { transform:translateX(200%); }
.sv-fact { font-size:14px; color:var(--ink); margin:var(--s-2) 0 0; }
.sv-fact .muted { margin-right:6px; }
.verdict .verdict-conf + .sv-fact { margin-top:var(--s-4); }
#sv-src { flex:1 1 180px !important; min-width:180px !important; }
.sv-trust { margin:var(--s-2) 0 0; max-width:75ch; }
.sv-details { margin-top:var(--s-4) !important; }
.sv-details .why { border-top:0; margin-top:0; padding-top:0; }
#sv-layer label { position:relative; z-index:1; justify-content:center; border:0 !important; border-radius:999px !important;
    padding:5px 16px !important; background:transparent !important; box-shadow:none !important; font-size:13px;
    font-weight:500; color:var(--ink-2); cursor:pointer; transition:color 200ms var(--ease); white-space:nowrap; }
#sv-layer label.selected { color:#fff !important; }
#sv-layer label span { margin:0 !important; }
#sv-layer input { position:absolute; opacity:0; pointer-events:none; }
#sv-layer label, #sv-layer label:focus-within { outline:none !important; }
#sv-layer .wrap:has(input:focus-visible) { box-shadow:0 0 0 3px oklch(0.42 0.12 255 / 0.25); }
#sv-sim .grid-container { grid-template-columns:repeat(5, minmax(0, 1fr)) !important; }
#sv-upload { white-space:nowrap; min-width:max-content !important; flex:0 0 auto !important; color:var(--ink) !important;
             border:1px solid var(--line) !important; background:var(--surface) !important; border-radius:8px !important; }
#sv-upload:hover { background:var(--hover) !important; }
@media (max-width: 640px) { .sv-controls { flex-wrap:wrap !important; } #sv-src { flex-basis:100% !important; } }
#sv-image img { width:100% !important; height:auto !important; border-radius:var(--radius); cursor:crosshair; }
#sv-image, #sv-patch { background:transparent !important; border:none !important; box-shadow:none !important; }
#sv-patch img { width:min(100%, 280px) !important; height:auto !important; border-radius:var(--radius); }
.sv-summary { margin-top:var(--s-3); }
.sv-count { font-size:14px; font-weight:500; color:var(--ink); }
.sv-legend { border-top:0 !important; padding-top:var(--s-2) !important; }
.sv-empty { min-height:260px; }

/* Lightbox for reference images */
.lightbox { border:0; padding:0; background:transparent; max-width:none; max-height:none; overflow:visible; }
.lightbox::backdrop { background:oklch(0.22 0.02 260 / 0.55); }
.lightbox[open] { animation:fade-up 220ms var(--ease); }
.lightbox figure { position:relative; margin:0; padding:var(--s-3); background:var(--surface); border-radius:var(--radius); }
.lightbox img { display:block; max-width:min(88vw, 1024px); max-height:76vh; width:auto; height:auto; border-radius:8px; }
.lightbox figcaption { font-size:14px; color:var(--ink-2); margin:var(--s-3) var(--s-1) var(--s-1); }
.lightbox-close { position:absolute; top:var(--s-5); right:var(--s-5); padding:6px 12px; border:0; border-radius:999px;
                  background:var(--surface); color:var(--ink); font:500 13px/1 inherit; cursor:pointer;
                  box-shadow:0 1px 4px oklch(0 0 0 / 0.18); }
.lightbox-close:hover { background:var(--hover); }
.compare .thumbnail-item { cursor:zoom-in; }

/* Motion: tab underline slides, new content fades in, data bars draw from the left */
[role=tablist] { position:relative; }
.tab-indicator { position:absolute; left:0; bottom:0; height:2px; border-radius:2px; background:var(--accent);
                 pointer-events:none; transition:transform 260ms var(--ease), width 260ms var(--ease), opacity 150ms; }
.has-indicator button[role=tab].selected::after { opacity:0 !important; }
button[role=tab] { color:var(--ink-2) !important; transition:color 200ms var(--ease) !important; }
button[role=tab]:hover { color:var(--ink) !important; }
button[role=tab].selected { color:var(--accent) !important; }
.gradio-container b, .gradio-container strong { color:inherit !important; }  /* Gradio's prose sets its own gray */
.gradio-container .caption-label { color:var(--ink) !important; }
.prob-head span:not(.prob-val), .hund-legend span { color:inherit !important; }
.gradio-container button.primary { border-radius:8px !important; }
@keyframes fade-up { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
@keyframes fade-in { from { opacity:0; } to { opacity:1; } }
@keyframes draw { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
.tabitem { animation:fade-in 220ms var(--ease); }
#lk-verdict .verdict, #lk-verdict .empty, .why, .next-steps, .lk-probs { animation:fade-up 260ms var(--ease); }
.prob-fill, .hund-bar { animation:draw 600ms var(--ease); }
details[open] > :not(summary) { animation:fade-up 220ms var(--ease); }
button.primary:active, button.secondary:active { transform:scale(0.98); }

/* Motion: state changes only, short, and off for reduced motion */
button, .thumbnail-item, #queue-table .body-cell { transition: background-color 150ms var(--ease), box-shadow 150ms var(--ease), transform 150ms var(--ease); }
.thumbnail-item:hover { transform: translateY(-1px); }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition:none !important; animation:none !important; } }
"""

# Workbench is light-only: force Gradio's light mode regardless of the OS setting.
FORCE_LIGHT = """() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}"""

# One underline per tab bar that slides to the selected tab (Gradio draws a static one per button).
TAB_INDICATOR = """() => {
  const place = (list) => {
    let bar = list.querySelector(':scope > .tab-indicator');
    const fresh = !bar;
    if (fresh) {
      bar = document.createElement('span');
      bar.className = 'tab-indicator';
      bar.style.transition = 'none';  // first placement: no slide in from the left edge
      list.appendChild(bar);
      list.classList.add('has-indicator');
    }
    const tab = list.querySelector('button[role=tab].selected');
    if (!tab) { bar.style.opacity = '0'; return; }
    bar.style.opacity = '1';
    bar.style.width = tab.offsetWidth + 'px';
    bar.style.transform = 'translateX(' + tab.offsetLeft + 'px)';
    if (fresh) requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.transition = ''; }));
  };
  let queued = false;
  const scan = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; document.querySelectorAll('[role=tablist]').forEach(place); });
  };
  new MutationObserver(scan).observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ['class']});
  window.addEventListener('resize', scan);
  scan();
}"""
# Click a reference image -> a large view in a native <dialog> (Esc, the backdrop or Close dismiss it).
LIGHTBOX = """() => {
  const box = document.createElement('dialog');
  box.className = 'lightbox';
  box.innerHTML = '<figure><button type="button" class="lightbox-close">Close</button><img alt="">'
                + '<figcaption></figcaption></figure>';
  document.body.appendChild(box);
  const img = box.querySelector('img'), cap = box.querySelector('figcaption');
  box.addEventListener('click', (e) => { if (e.target === box || e.target.closest('.lightbox-close')) box.close(); });
  document.addEventListener('click', (e) => {
    const item = e.target.closest('.compare .thumbnail-item');
    const pic = item && item.querySelector('img');
    if (!pic) return;
    e.preventDefault();
    e.stopPropagation();
    const group = item.closest('.compare').querySelector('[data-testid=block-label]');
    const label = item.querySelector('.caption-label');
    cap.textContent = [group && group.innerText, label && label.innerText].filter(Boolean).join(' \u00b7 ');
    img.alt = cap.textContent;
    img.src = pic.currentSrc || pic.src;
    box.showModal();
  }, true);
}"""
JS = f"() => {{ ({FORCE_LIGHT})(); ({TAB_INDICATOR})(); ({LIGHTBOX})(); }}"


with gr.Blocks(title="Pathology Assistant") as demo:
    if not args.embedded:
        gr.HTML(app_header_html())
    decisions = gr.State({})  # this browser session's sign-offs

    # Queue and Review are hidden for now (Bowen, 10/17); --show-review brings them back.
    SHOW_REVIEW = args.show_review or not LIVE
    with gr.Tabs(selected="queue" if SHOW_REVIEW else ("slides" if SLIDE_VIEW else "lookup")) as tabs:
        # A. Pre-read / triage: the doctor only sees what needs a doctor.
        with gr.Tab("Queue", id="queue", visible=SHOW_REVIEW):
            with gr.Row(equal_height=True):
                queue_head = gr.HTML(queue_head_html({}))
            with gr.Row(equal_height=True):
                start_btn = start_button({})
                view = gr.Radio(VIEWS, value=VIEWS[0], show_label=False, container=False, scale=3)
            with gr.Row(equal_height=False, elem_classes="cleared-row"):
                cleared_note = gr.HTML(cleared_note_html(), visible=False, scale=3)
                spot_btn = gr.Button("Spot-check a random tile", variant="secondary", visible=False, scale=1)
            queue_df = gr.Dataframe(
                value=queue_table(VIEWS[0], {}), interactive=False, elem_id="queue-table",
                datatype=["html", "str", "str", "number", "str", "html"],
            )

        with gr.Tab("Review", id="case", visible=SHOW_REVIEW):
            # Holds the open case; navigation happens through the queue and "next", so it stays hidden.
            case_picker = gr.Dropdown(
                choices=[c["case_id"] for c in ORDERED], value=(FLAGGED or ORDERED)[0]["case_id"],
                label="Case", visible=False,
            )
            with gr.Row(equal_height=False, elem_classes="toolbar"):
                case_header = gr.HTML(scale=5)
                show_truth = gr.Checkbox(label="Demo: show ground truth", value=False, container=False,
                                         scale=0, min_width=210, elem_id="truth-toggle")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    case_image = gr.Image(show_label=False, buttons=[], elem_id="case-image")
                with gr.Column(scale=6):
                    verdict = gr.HTML(elem_id="verdict", elem_classes="flush")
                    with gr.Row(elem_classes="actions"):
                        accept_btn = gr.Button("Accept and next", variant="primary", scale=0, min_width=170)
                        correct_open = gr.Button("Correct…", variant="secondary", scale=0, min_width=120)
                        skip_btn = gr.Button("Skip for now", variant="secondary", scale=0, min_width=120,
                                             elem_classes="link-btn")
                    # Shown only after "Correct…"; preselected with the runner-up candidate.
                    with gr.Row(visible=False, equal_height=True, elem_classes="correction") as correction_row:
                        new_label = gr.Dropdown(choices=[(full_name(c), c) for c in CLASSES], value=None,
                                                show_label=False, container=False, scale=3)
                        correct_btn = gr.Button("Save and next", variant="secondary", scale=0, min_width=140)
                        cancel_btn = gr.Button("Cancel", variant="secondary", scale=0, min_width=90,
                                               elem_classes="link-btn")
                    case_why = gr.HTML(elem_classes="flush")
            gr.HTML(f'<h3 class="subhead">Compare with confirmed reference {REF_NOUN}</h3>{SIMILARITY_NOTE}',
                    elem_classes="flush")
            with gr.Row():
                case_compare = [gr.Gallery(columns=3, height=200, visible=False, buttons=[], allow_preview=False, elem_classes="compare")
                                for _ in range(N_COMPARE)]
            case_refs = gr.HTML(visible=False, elem_classes="flush")
            with gr.Accordion(f"More similar confirmed {REF_NOUN}, any class", open=False):
                case_neighbors = gr.Gallery(show_label=False, columns=5, height=255, buttons=[], allow_preview=False, elem_classes="compare")

        # A2. Slide view: look over a whole slide (or a large photo) first, then click into a patch
        if SLIDE_VIEW:
            with gr.Tab("Slide view", id="slides"):
                gr.HTML('<h2 class="page-title">Look over a slide</h2><p class="lede">Colors mark where to look first. '
                        "Click a spot to inspect that patch.</p>", elem_classes="flush")
                first = sv_choices()[0]
                e0 = sv_source(first)
                with gr.Row(equal_height=False, elem_id="sv-row"):
                    with gr.Column(scale=7, elem_id="sv-left"):
                        with gr.Row(equal_height=True, elem_classes="sv-controls"):
                            sv_src = gr.Dropdown(sv_choices(), value=first, show_label=False, container=False, scale=3,
                                                 elem_id="sv-src")
                            sv_lay = gr.Radio(sv_layers(e0), value="Suspicious", show_label=False,
                                              container=False, elem_id="sv-layer", scale=3)
                            sv_up = gr.UploadButton("Upload photo", file_types=["image"], variant="secondary",
                                                    size="sm", scale=0, elem_id="sv-upload")
                        sv_img = gr.Image(value=sv_render(e0, "Suspicious"), interactive=False,
                                          show_label=False, buttons=[], elem_id="sv-image")
                        sv_sum = gr.HTML(sv_summary_html(e0, "Suspicious"), elem_classes="flush")
                    with gr.Column(scale=5, elem_id="sv-right"):
                        sv_patch = gr.Image(visible=False, interactive=False, show_label=False, buttons=[],
                                            elem_id="sv-patch")
                        sv_verdict = gr.HTML(SV_EMPTY, elem_classes="flush")
                        sv_why = gr.HTML(visible=False, elem_classes="flush")
                        sv_sim = gr.Gallery(visible=False, columns=5, height=120, buttons=[], allow_preview=False,
                                            label="Most similar confirmed patches", elem_classes="compare",
                                            elem_id="sv-sim")
                sv_state = gr.State({"choice": first, "sel": None})
            sv_outputs = [sv_img, sv_sum, sv_state, sv_patch, sv_verdict, sv_why, sv_sim, sv_lay]
            # .input: only a pick by the user; an upload sets the menu itself
            sv_src.input(fn=sv_show, inputs=[sv_src, sv_lay], outputs=sv_outputs, api_visibility="private")
            sv_up.upload(fn=sv_upload, inputs=[sv_up, sv_lay, sv_state], outputs=[*sv_outputs, sv_src],
                         api_visibility="private")
            sv_lay.input(fn=sv_relayer, inputs=[sv_lay, sv_state], outputs=[sv_img, sv_sum], api_visibility="private")
            sv_img.select(fn=sv_click, inputs=[sv_lay, sv_state],
                          outputs=[sv_img, sv_state, sv_patch, sv_verdict, sv_why, sv_sim], api_visibility="private")

        # B. Pathology dictionary
        if LIVE:
            with gr.Tab("Lookup", id="lookup"):
                hint = escape(META.get("input_hint", "Upload one H&E tile (about 224×224 px at 0.5 µm/px)"))
                gr.HTML(f'<h2 class="page-title">Look up {"an image" if REF_NOUN == "images" else "a tile"}</h2>'
                        f'<p class="lede">{hint} to compare it with confirmed reference {REF_NOUN}.</p>'
                        '<p class="small muted privacy">Public research demo: please do not upload images that could '
                        "identify a patient.</p>",
                        elem_classes="flush")
                # Created first so the examples strip can sit above it; placed in the row below.
                lk_image = gr.Image(type="pil", show_label=False, elem_id="lookup-image",
                                    sources=["upload", "clipboard"], render=False)
                example_tiles = sorted(p for p in (DATA_DIR / "examples").glob("*") if p.suffix in (".png", ".jpg", ".webp"))[:6]
                with gr.Row(equal_height=False):
                    # Input column: examples sit above the upload, so the result column starts at the top
                    with gr.Column(scale=5, elem_id="lk-left"):
                        if example_tiles:
                            gr.Examples(examples=[[str(p)] for p in example_tiles], inputs=lk_image,
                                        label=META.get("examples_label", "Try an example (not in the queue)"),
                                        examples_per_page=6)
                        lk_image.render()
                        lk_btn = gr.Button("Look up", variant="primary")
                        # Why sits under the input so the two columns end at about the same height
                        lk_why = gr.HTML(visible=False, elem_classes="flush")
                    with gr.Column(scale=6):
                        lk_verdict = gr.HTML(LOOKUP_EMPTY, elem_id="lk-verdict", elem_classes="flush")
                lk_compare_title = gr.HTML(f'<h3 class="subhead">Compare with confirmed reference {REF_NOUN}</h3>'
                                           f'{SIMILARITY_NOTE}',
                                           visible=False, elem_classes="flush")
                with gr.Row():
                    lk_compare = [gr.Gallery(columns=3, height=200, visible=False, buttons=[], allow_preview=False, elem_classes="compare")
                                  for _ in range(N_COMPARE)]
                lk_refs = gr.HTML(visible=False, elem_classes="flush")
                with gr.Accordion(f"More similar confirmed {REF_NOUN}, any class", open=False, visible=False) as lk_more:
                    lk_neighbors = gr.Gallery(show_label=False, columns=5, height=255, buttons=[], allow_preview=False, elem_classes="compare")
            lk_btn.click(fn=lookup, inputs=lk_image,
                         outputs=[lk_verdict, lk_why, lk_compare_title, *lk_compare, lk_refs, lk_more, lk_neighbors])

        # C. Evidence
        with gr.Tab("Evidence", id="evidence"):
            if EVIDENCE is None:
                gr.HTML('<div class="placeholder">No evidence data yet. Add <code>evidence.json</code> '
                        "to the data folder (format: docs/INTERFACE.md).</div>")
            else:
                gr.HTML(evidence_head_html(EVIDENCE), elem_classes="flush")
                if LABELED:
                    gr.Markdown("### Does “90% sure” mean 90% right?", elem_classes="section-title")
                    gr.HTML(calibration_html(LABELED), elem_classes="flush")
                if LABELED:
                    gr.Markdown("### How much can the AI handle on its own?", elem_classes="section-title")
                    gr.HTML(per_hundred_html(LABELED), elem_classes="flush")
                if LABELED and all(len(c["similar_cases"]) == 5 for c in LABELED):
                    gr.Markdown("### When the most similar images disagree", elem_classes="section-title")
                    gr.HTML(agreement_html(LABELED), elem_classes="flush")
                gr.Markdown(f"### {EVIDENCE.get('models_title', 'Model comparison')}", elem_classes="section-title")
                gr.Markdown("Each row is one way of deciding which images go to a pathologist.", elem_classes="ev-md")
                gr.Dataframe(value=model_table(EVIDENCE["models"]), interactive=False, buttons=[], wrap=True)
                if EVIDENCE.get("by_patient"):
                    gr.Markdown("### Errors by patient", elem_classes="section-title")
                    gr.Markdown(EVIDENCE.get("by_patient_note", ""), elem_classes="ev-md")
                    gr.Dataframe(value=by_patient_table(EVIDENCE["by_patient"]), interactive=False, buttons=[],
                                 wrap=True, max_height=420)
                if EVIDENCE.get("robustness"):
                    rb = EVIDENCE["robustness"]
                    gr.Markdown("### When the input changes", elem_classes="section-title")
                    gr.Markdown(f"{robustness_sentence(rb)} {rb['source']}", elem_classes="ev-md")
                    gr.Dataframe(value=robustness_table(rb), interactive=False, buttons=[], wrap=True)
                if SLIDE_VALIDATION:
                    gr.Markdown("### Slide maps against expert annotations", elem_classes="section-title")
                    gr.Markdown(f"{SLIDE_VALIDATION_NOTE} {slide_validation_sentence(SLIDE_VALIDATION)}",
                                elem_classes="ev-md")
                    gr.Dataframe(value=slide_validation_table(SLIDE_VALIDATION), interactive=False, buttons=[], wrap=True)
                gr.Markdown("### About this data", elem_classes="section-title")
                gr.Markdown(EVIDENCE["split_note"], elem_classes="ev-md")

    gr.HTML(f'<footer class="app-footer">{META["attribution"]}</footer>' if META.get("attribution") else APP_FOOTER,
            elem_classes="flush")

    case_outputs = [case_header, case_image, verdict, *case_compare, case_refs, case_why, case_neighbors,
                    new_label, correction_row]
    decision_outputs = [case_picker, case_header, queue_df, queue_head, start_btn, decisions]

    # Queue
    view.change(fn=change_view, inputs=[view, decisions], outputs=[queue_df, cleared_note, spot_btn],
                api_visibility="private")
    queue_df.select(fn=open_case, inputs=None, outputs=[case_picker, tabs])
    start_btn.click(fn=start_review, inputs=decisions, outputs=[case_picker, tabs])
    spot_btn.click(fn=spot_check, inputs=None, outputs=[case_picker, tabs])
    # Review
    case_picker.change(fn=show_case, inputs=[case_picker, show_truth, decisions], outputs=case_outputs)
    show_truth.change(fn=show_case, inputs=[case_picker, show_truth, decisions], outputs=case_outputs,
                      api_visibility="private")
    accept_btn.click(fn=accept, inputs=[case_picker, view, decisions], outputs=decision_outputs)
    correct_btn.click(fn=correct, inputs=[case_picker, new_label, view, decisions], outputs=decision_outputs)
    skip_btn.click(fn=skip, inputs=[case_picker, decisions], outputs=case_picker)
    correct_open.click(fn=lambda: gr.Row(visible=True), outputs=correction_row, api_visibility="private")
    cancel_btn.click(fn=lambda: gr.Row(visible=False), outputs=correction_row, api_visibility="private")

    # Page load: start this session from the saved sign-offs (none in session mode), then draw everything.
    def on_load(case_id, show_truth, view):
        decided = load_decisions()
        return (decided, *show_case(case_id, show_truth, decided), *queue_outputs(view, decided))

    demo.load(fn=on_load, inputs=[case_picker, show_truth, view],
              outputs=[decisions, *case_outputs, queue_df, queue_head, start_btn], api_visibility="private")


if __name__ == "__main__":
    demo.launch(
        theme=THEME, css=CSS, js=JS, footer_links=[], share=args.share,
        favicon_path=str(Path(__file__).resolve().parent / "assets/favicon.png"),
        allowed_paths=[str(DATA_DIR), str(LIVE_CACHE)],
    )
