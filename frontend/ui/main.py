"""Pathology Assistant with a dataset switch: one page, one app.py instance per dataset.

Each dataset keeps its own classifier, thresholds, gallery and Evidence page: the datasets cannot be mixed
(BACH and BreakHis embeddings are fully separable; a classifier trained on one fails on the other).
app.py is loaded once per data folder as its own module, so each copy has its own globals; this page draws
the header and a switch, and shows one copy at a time.

Run: python main.py --data data_bach data_breakhis --live --session-decisions [--share]
"""
import argparse
import importlib.util
import sys
from html import escape
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=["data_bach", "data_breakhis"], help="data folders under ui/")
ap.add_argument("--live", action="store_true", help="enable Lookup (loads QuiltNet once per dataset)")
ap.add_argument("--session-decisions", action="store_true", help="sign-offs per browser session only (public demo)")
ap.add_argument("--share", action="store_true", help="also create a temporary public *.gradio.live link")
args = ap.parse_args()


def load_app(data: str, extra: list[str]):
    """Import app.py as a separate module for one data folder (it reads its options from sys.argv)."""
    argv = sys.argv
    sys.argv = ["app.py", "--data", data, "--embedded", *extra]
    try:
        spec = importlib.util.spec_from_file_location(f"app_{data}", HERE / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


extra = (["--live"] if args.live else []) + (["--session-decisions"] if args.session_decisions else [])
apps = [load_app(d, extra) for d in args.data]
labels = [a.META.get("dataset_label") or a.META.get("source", d).split(" ")[0] for a, d in zip(apps, args.data)]

SWITCH_CSS = f"#dataset-switch {{ --n: {len(labels)}; }}" + "".join(  # pill position per selected option
    f"#dataset-switch .wrap:has(> label:nth-child({k + 1}).selected)::before {{ transform:translateX({100 * k}%); }}"
    for k in range(len(labels)))

SITE_CSS = """
.site-header { align-items:flex-end !important; justify-content:space-between !important; gap:var(--s-4) !important;
               padding:8px 2px 16px; border-bottom:1px solid var(--line); flex-wrap:nowrap !important; }
#site-title { flex:1 1 auto !important; min-width:0 !important; }
#dataset-switch { margin-left:auto; }
@media (max-width: 640px) { .site-header { flex-wrap:wrap !important; } #dataset-switch { margin-left:0; } }
.dataset-line { font-size:13px; color:var(--muted); margin:var(--s-3) 0 0; }
/* Dataset switch: a segmented control, current dataset in the accent color */
#dataset-switch { flex:0 0 auto !important; min-width:0 !important; width:max-content !important;
                  overflow:visible !important; }  /* Gradio sets overflow:auto, which would clip the focus halo */
/* Segmented control: equal segments, one accent pill that slides under the selected one */
#dataset-switch .wrap:has(> label) { position:relative; display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:0;
    padding:3px; border:1px solid var(--line); border-radius:999px; background:var(--surface); }
#dataset-switch .wrap:has(> label)::before { content:""; position:absolute; top:3px; bottom:3px; left:3px;
    width:calc((100% - 6px) / var(--n)); border-radius:999px; background:var(--accent);
    transition:transform 280ms var(--ease); }
#dataset-switch label { position:relative; z-index:1; justify-content:center; border:0 !important; border-radius:999px !important;
    padding:5px 16px !important; background:transparent !important; box-shadow:none !important; font-size:13px;
    font-weight:500; color:var(--ink-2); cursor:pointer; transition:color 200ms var(--ease); }
#dataset-switch label.selected { color:#fff !important; }
#dataset-switch input { position:absolute; opacity:0; pointer-events:none; }
#dataset-switch label span { margin:0 !important; }  /* Gradio leaves room for a radio dot; text must sit centered */
#dataset-switch label, #dataset-switch label:focus-within { outline:none !important; }
/* Keyboard focus: a soft halo around the whole control (not a second ring around the pill) */
#dataset-switch .wrap:has(input:focus-visible) { box-shadow:0 0 0 3px oklch(0.42 0.12 255 / 0.25); }
/* Switching datasets: the new one fades in instead of snapping into place */
.dataset-col { animation:fade-up 280ms var(--ease); }
"""

with gr.Blocks(title="Pathology Assistant") as site:
    with gr.Row(elem_classes="site-header", equal_height=False):
        gr.HTML('<div><h1 class="app-title">Pathology Assistant</h1>'
                '<p class="app-subtitle">AI pre-read and reference lookup for H&amp;E histology</p></div>',
                elem_classes="flush", elem_id="site-title")
        switch = gr.Radio(labels, value=labels[0], show_label=False, container=False, elem_id="dataset-switch",
                          scale=0)
    columns = []
    for i, app in enumerate(apps):
        with gr.Column(visible=i == 0, elem_classes="dataset-col") as column:
            gr.HTML(f'<p class="dataset-line">{escape(app.META.get("dataset", ""))} · '
                    f'{escape(app.META.get("model", ""))}</p>', elem_classes="flush")
            app.demo.render()
        columns.append(column)
    switch.change(fn=lambda chosen: [gr.Column(visible=chosen == label) for label in labels],
                  inputs=switch, outputs=columns, api_visibility="private")

def launch_options() -> dict:
    """Theme, CSS, JS and file access for this page; shared by local launch and deploy/modal_app.py."""
    first = apps[0]
    return dict(theme=first.THEME, css=first.CSS + SITE_CSS + SWITCH_CSS, js=first.JS, footer_links=[],
                allowed_paths=[p for a in apps for p in (str(a.DATA_DIR), str(a.LIVE_CACHE))])


if __name__ == "__main__":
    site.launch(share=args.share, **launch_options())
