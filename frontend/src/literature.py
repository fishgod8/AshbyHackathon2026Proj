"""Retrieve real PubMed papers for each differential (pair of candidate classes).

Evidence, not conclusions: for every pair we store the exact PubMed query and the
title, first author, journal, year and link of the papers it returns. No text is
generated or summarized. Every paper starts as "reviewed": false; the team
(ideally a pathologist) flips it after checking, and the UI says so.

Two searches per pair:
  differential  papers on telling the two apart (histology / differential diagnosis)
  ihc           papers on immunohistochemistry for the same pair

Search words per class come from a terms file ({code: ["term", ...]}; an empty list
skips that class, e.g. background). Without one, the class names in meta.json are used.

Example:
    python -m src.literature --data ui/data_semi --terms ui/literature_terms_crc.json \
        --context "colorectal OR colon OR rectal"
    python -m src.literature --data ui/data_semi --mark 22507525     # a person checked this paper
    python -m src.literature --data ui/data_semi --unmark 22507525
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# Humans, including records not yet MeSH-indexed (the standard PubMed idiom).
HUMANS_ONLY = "NOT (animals[mh] NOT humans[mh])"
TOPICS = {
    "differential": '(histolog*[tiab] OR morpholog*[tiab] OR "differential diagnosis"[tiab])',
    "ihc": "(immunohistochem*[tiab] OR IHC[tiab])",
}
REQUEST_GAP_S = 0.4  # NCBI allows ~3 requests/s without an API key


def eutils(path: str, **params) -> dict:
    params.update(tool="ashby-hackathon", retmode="json")
    url = EUTILS + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                time.sleep(REQUEST_GAP_S)
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def class_clause(terms: list[str]) -> str:
    return "(" + " OR ".join(f'"{t}"[tiab]' for t in terms) + ")"


def build_query(a: list[str], b: list[str], topic: str, context: str | None) -> str:
    parts = [class_clause(a), class_clause(b)]
    if context:
        parts.append(f"({' OR '.join(f'{w}[tiab]' for w in context.split(' OR '))})")
    parts.append(TOPICS[topic])
    return " AND ".join(parts) + " " + HUMANS_ONLY


def clean_title(title: str) -> str:
    return re.sub(r"<[^>]+>", "", title).strip()


def search(query: str, n: int) -> tuple[int, list[dict]]:
    found = eutils("esearch.fcgi", db="pubmed", term=query, retmax=n, sort="relevance")["esearchresult"]
    ids = found["idlist"]
    if not ids:
        return int(found["count"]), []
    summary = eutils("esummary.fcgi", db="pubmed", id=",".join(ids))["result"]
    papers = []
    for pmid in ids:
        rec = summary[pmid]
        authors = rec.get("authors") or []
        papers.append({
            "pmid": pmid,
            "title": clean_title(rec.get("title", "")),
            "first_author": authors[0]["name"] if authors else "",
            "journal": rec.get("source", ""),
            "year": rec.get("pubdate", "")[:4],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "reviewed": False,
        })
    return int(found["count"]), papers


def search_url(query: str) -> str:
    """The same search on the PubMed website, so anyone can check it."""
    return "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": query})


def retrieve(a: list[str], b: list[str], topic: str, context: str | None, n: int) -> dict:
    """Search with the context words; if nothing comes back, retry without them and say so."""
    query = build_query(a, b, topic, context)
    count, papers = search(query, n)
    relaxed = False
    if not papers and context:
        query, relaxed = build_query(a, b, topic, None), True
        count, papers = search(query, n)
    return {"query": query, "url": search_url(query), "relaxed": relaxed, "count": count, "papers": papers}


def reviewed_marks(path: Path) -> set[tuple[str, str, str]]:
    """(pair, topic, pmid) a person already marked, so a re-run keeps their review."""
    if not path.exists():
        return set()
    old = json.loads(path.read_text())
    return {(k, t, p["pmid"]) for k, pair in old.items() if not k.startswith("_")
            for t, entry in pair.items() for p in entry["papers"] if p.get("reviewed")}


def set_reviewed(path: Path, pmids: set[str], value: bool) -> int:
    """Mark papers as checked (or not) by a person, in every pair they appear in."""
    lit = json.loads(path.read_text())
    changed = 0
    for key, pair in lit.items():
        if key.startswith("_"):
            continue
        for topic in pair.values():
            for paper in topic["papers"]:
                if paper["pmid"] in pmids and paper["reviewed"] != value:
                    paper["reviewed"], changed = value, changed + 1
    path.write_text(json.dumps(lit, indent=2, ensure_ascii=False))
    return changed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="UI data folder with meta.json; output goes here")
    p.add_argument("--mark", nargs="+", default=None, metavar="PMID", help="mark papers as reviewed, then exit")
    p.add_argument("--unmark", nargs="+", default=None, metavar="PMID", help="mark papers as not reviewed, then exit")
    p.add_argument("--terms", type=Path, default=None, help="JSON {class code: [search terms]}")
    p.add_argument("--context", default=None, help='words every paper must mention, e.g. "colorectal OR colon"')
    p.add_argument("--per-pair", type=int, default=20, help="candidates per search (AI screening keeps the relevant ones)")
    args = p.parse_args()

    if args.mark or args.unmark:
        pmids, value = set(args.mark or args.unmark), bool(args.mark)
        n = set_reviewed(args.data / "literature.json", pmids, value)
        print(f"{'marked' if value else 'unmarked'} {n} paper entries")
        return

    meta = json.loads((args.data / "meta.json").read_text())
    names = meta.get("class_names", {})
    terms = json.loads(args.terms.read_text()) if args.terms else {c: [names.get(c, c)] for c in meta["classes"]}
    classes = [c for c in meta["classes"] if terms.get(c)]

    out = {"_meta": {
        "source": "PubMed (NCBI E-utilities), relevance order",
        "retrieved": date.today().isoformat(),
        "note": "Auto-retrieved. Not reviewed by a pathologist unless a paper says reviewed: true.",
    }}
    out_path = args.data / "literature.json"
    keep = reviewed_marks(out_path)
    for a, b in itertools.combinations(sorted(classes), 2):
        key = f"{a}|{b}"
        out[key] = {topic: retrieve(terms[a], terms[b], topic, args.context, args.per_pair) for topic in TOPICS}
        for topic, entry in out[key].items():
            for paper in entry["papers"]:
                paper["reviewed"] = (key, topic, paper["pmid"]) in keep
        print(f"{key:10s} differential {out[key]['differential']['count']:>5}  ihc {out[key]['ihc']['count']:>5}"
              + ("  (relaxed)" if out[key]["differential"]["relaxed"] or out[key]["ihc"]["relaxed"] else ""), flush=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out) - 1} pairs to {out_path} (kept {len(keep)} reviewed marks)")


if __name__ == "__main__":
    main()
