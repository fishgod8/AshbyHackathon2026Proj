"""AI screening of retrieved PubMed papers, with verbatim, machine-checked quotes.

For every differential pair and topic in <data>/literature.json, an LLM reads the
title + abstract of each candidate paper and answers two things:
  relevant  does this abstract help tell A from B (or report stains that do)?
  quote     ONE sentence copied verbatim from the abstract that shows why

The quote is then checked in code: it must appear in the abstract (after
whitespace/quote-mark normalization), otherwise it is dropped and the paper is not
shown as AI-screened. The model never writes text that reaches the UI; only
verified quotes from the abstract do. A pathologist's review (--mark in
src/literature.py) still outranks the AI screen.

Only public paper titles/abstracts are sent to the API. No images, no patient data.

Key: DEEPSEEK_API_KEY in the environment or in <repo>/.env (git-ignored; never printed).

Example:
    python -m src.screen_literature --data ui/data_semi
    python -m src.screen_literature --data ui/data_semi --dry-run   # no API calls, shows the plan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
API_URL = "https://api.deepseek.com/chat/completions"
EUTILS_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOPIC_ASK = {
    "differential": "help a pathologist tell {a} from {b} on H&E histology (morphologic features or differential diagnosis)",
    "ihc": "report immunohistochemical stains that help tell {a} from {b}",
}
MAX_ABSTRACT_CHARS = 2500


def api_key() -> str:
    """From the environment, else from <repo>/.env. The value is never printed."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key or key == "paste-your-key-here":
        raise SystemExit(f"DEEPSEEK_API_KEY not set. Paste your key into {env} (replace the placeholder).")
    return key


# ---- PubMed abstracts ----

def fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    """PMID -> abstract text (all labelled parts joined), 200 per request."""
    out: dict[str, str] = {}
    for i in range(0, len(pmids), 200):
        params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids[i:i + 200]),
                                         "retmode": "xml", "tool": "ashby-hackathon"})
        with urllib.request.urlopen(f"{EUTILS_FETCH}?{params}", timeout=60) as resp:
            root = ET.fromstring(resp.read())
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext("MedlineCitation/PMID") or ""
            parts = ["".join(node.itertext()).strip() for node in art.iter("AbstractText")]
            out[pmid] = " ".join(p for p in parts if p)
        time.sleep(0.4)
    return out


# ---- Quote check ----

def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def verified(quote: str, abstract: str) -> bool:
    """The quote must be whole sentence(s) of the abstract, verbatim.

    A fragment is refused even if its words appear: cutting a sentence (e.g. dropping
    a leading "not") can change its meaning.
    """
    q, a = normalize(quote).rstrip(".?!"), normalize(abstract)
    if not 30 <= len(q) <= 600:
        return False
    i = a.find(q)
    while i != -1:
        starts_sentence = i == 0 or (a[i - 1] == " " and i >= 2 and a[i - 2] in ".?!:")
        end = i + len(q)
        ends_sentence = end == len(a) or a[end] in ".?!"
        if starts_sentence and ends_sentence:
            return True
        i = a.find(q, i + 1)
    return False


# ---- LLM screening ----

def build_messages(a: str, b: str, topic: str, papers: list[dict]) -> list[dict]:
    items = "\n\n".join(
        f"PMID {p['pmid']}\nTitle: {p['title']}\nAbstract: {p['abstract'][:MAX_ABSTRACT_CHARS] or '(no abstract)'}"
        for p in papers
    )
    ask = TOPIC_ASK[topic].format(a=a, b=b)
    return [
        {"role": "system", "content": "You screen PubMed abstracts for a pathology reference tool. "
                                      "You answer in json only and never invent text."},
        {"role": "user", "content": (
            f"Differential: {a} vs {b}.\n"
            f"Keep only papers whose abstract itself would {ask}, in human tissue. "
            "Papers that only mention both terms (prognosis, drug targets, animal models, other organs, "
            "unrelated techniques) are not relevant.\n"
            "For each relevant paper, copy exactly ONE sentence from its abstract, character for character, "
            "that shows why it is relevant. Do not paraphrase, shorten or merge sentences. "
            "If no single sentence shows it, mark the paper not relevant.\n\n"
            'Return json like {"papers": [{"pmid": "12345", "relevant": true, "quote": "..."}, '
            '{"pmid": "67890", "relevant": false, "quote": ""}]} with one entry per paper below.\n\n'
            f"{items}"
        )},
    ]


# deepseek-flash reasons before answering and its reasoning tokens count toward max_tokens
# (one 20-paper call used ~1,900), so leave plenty of room or the answer comes back empty.
MAX_TOKENS = 12000


def call_llm(messages: list[dict], key: str, model: str) -> tuple[dict | None, str]:
    """Return (parsed json, "") or (None, what went wrong) after retries."""
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS,
                       "response_format": {"type": "json_object"}, "stream": False}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    problem = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                choice = json.load(resp)["choices"][0]
            content = choice["message"].get("content") or ""
            if content.strip():
                return json.loads(content), ""
            problem = f"empty answer (finish_reason={choice.get('finish_reason')})"
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):
                raise SystemExit(f"DeepSeek API error {e.code} (check the key or account balance)")
            problem = f"HTTP {e.code}"
        except json.JSONDecodeError:
            problem = "malformed json"
        except (urllib.error.URLError, TimeoutError) as e:
            problem = f"network: {e}"
        time.sleep(2 ** attempt)
    return None, problem


def apply_screen(papers: list[dict], answer: dict, abstracts: dict[str, str]) -> tuple[int, int]:
    """Attach the screen to each paper; return (judged relevant, relevant with a verified quote)."""
    by_pmid = {str(r.get("pmid")): r for r in answer.get("papers", [])}
    relevant = shown = 0
    for p in papers:
        r = by_pmid.get(p["pmid"], {})
        quote = (r.get("quote") or "").strip()
        ok = bool(r.get("relevant")) and verified(quote, abstracts.get(p["pmid"], ""))
        relevant += bool(r.get("relevant"))
        shown += ok
        p["ai"] = {"relevant": bool(r.get("relevant")), "quote": quote if ok else "", "verified": ok}
    return relevant, shown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="UI data folder holding literature.json and meta.json")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--dry-run", action="store_true", help="fetch abstracts and show the plan; no API calls")
    ap.add_argument("--force", action="store_true", help="re-screen pairs already screened with this model")
    args = ap.parse_args()

    path = args.data / "literature.json"
    lit = json.loads(path.read_text())
    names = json.loads((args.data / "meta.json").read_text()).get("class_names", {})
    jobs = [(k, t, e) for k, pair in lit.items() if not k.startswith("_") for t, e in pair.items() if e["papers"]]
    pmids = sorted({p["pmid"] for _, _, e in jobs for p in e["papers"]})
    print(f"{len(jobs)} screening calls, {len(pmids)} unique papers; fetching abstracts...", flush=True)
    abstracts = fetch_abstracts(pmids)
    print(f"abstracts found for {sum(bool(v) for v in abstracts.values())} of {len(pmids)}", flush=True)
    if args.dry_run:
        return

    key = api_key()
    total_rel = total_shown = 0
    failed = []
    for i, (k, topic, entry) in enumerate(jobs, start=1):
        done = entry.get("screening", {})
        if done.get("model") == args.model and "error" not in done and not args.force:
            print(f"[{i}/{len(jobs)}] {k:10s} {topic:12s} already screened, skipped", flush=True)
            continue
        a, b = (names.get(c, c) for c in k.split("|"))
        papers = [{**p, "abstract": abstracts.get(p["pmid"], "")} for p in entry["papers"]]
        answer, problem = call_llm(build_messages(a, b, topic, papers), key, args.model)
        if answer is None:
            entry["screening"] = {"model": args.model, "date": date.today().isoformat(), "error": problem}
            failed.append(f"{k} {topic}")
            print(f"[{i}/{len(jobs)}] {k:10s} {topic:12s} FAILED ({problem}); rerun to retry", flush=True)
        else:
            rel, shown = apply_screen(entry["papers"], answer, abstracts)
            entry["screening"] = {"model": args.model, "date": date.today().isoformat()}
            total_rel, total_shown = total_rel + rel, total_shown + shown
            print(f"[{i}/{len(jobs)}] {k:10s} {topic:12s} relevant {rel:2d}, verified quote {shown:2d} of {len(papers)}", flush=True)
        path.write_text(json.dumps(lit, indent=2, ensure_ascii=False))  # checkpoint: safe to interrupt
    lit["_meta"]["ai_screening"] = {"model": args.model, "date": date.today().isoformat(),
                                    "note": "Only papers with a quote verified against the abstract are shown as AI-screened."}
    path.write_text(json.dumps(lit, indent=2, ensure_ascii=False))
    print(f"done: {total_rel} judged relevant, {total_shown} with a verified quote this run"
          + (f"; {len(failed)} failed: {', '.join(failed)} (rerun to retry)" if failed else ""))


if __name__ == "__main__":
    main()
