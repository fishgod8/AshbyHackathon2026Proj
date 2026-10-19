import time
from playwright.sync_api import sync_playwright
JS_DONE = """() => { const v=[...document.querySelectorAll('#lk-verdict')].find(e=>e.offsetParent); return v && v.innerText.includes('confidence'); }"""
STRUCT = r"""() => {
  const vis = e => e.offsetParent !== null && e.getBoundingClientRect().height > 2;
  const radii = {}, icons = {}, lefts = {};
  for (const e of document.querySelectorAll('.gradio-container *')) {
    if (!vis(e)) continue;
    const s = getComputedStyle(e);
    const boxed = s.borderTopWidth !== '0px' && s.borderTopStyle !== 'none' || (s.backgroundColor !== 'rgba(0, 0, 0, 0)' && e.getBoundingClientRect().width > 40);
    if (boxed && s.borderRadius !== '0px') { const k = s.borderRadius; (radii[k] = radii[k] || new Set()).add(e.tagName.toLowerCase() + '.' + String(e.className).split(' ').filter(c=>c && !c.startsWith('svelte')).slice(0,2).join('.')); }
    if (e.tagName === 'svg') { const r = e.getBoundingClientRect(); const k = Math.round(r.width) + 'x' + Math.round(r.height); (icons[k] = icons[k] || new Set()).add(String(e.parentElement.className).split(' ').filter(c=>c && !c.startsWith('svelte'))[0] || e.parentElement.tagName); }
  }
  // left edge of every section-level block inside the visible tab
  const tab = [...document.querySelectorAll('.tabitem')].find(vis);
  for (const e of tab.querySelectorAll(':scope .block, :scope .row, :scope h2, :scope h3, :scope .references, :scope .refs-fold, :scope .hund, :scope .ev-bars')) {
    if (!vis(e)) continue; const l = Math.round(e.getBoundingClientRect().left); if (l > 400) continue;
    (lefts[l] = lefts[l] || new Set()).add(e.tagName.toLowerCase() + '.' + String(e.className).split(' ').filter(c=>c && !c.startsWith('svelte')).slice(0,2).join('.'));
  }
  const fmt = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, [...v].slice(0, 6)]));
  return {radii: fmt(radii), icons: fmt(icons), lefts: fmt(lefts)};
}"""
with sync_playwright() as p:
    b = p.chromium.launch(); page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.goto("http://127.0.0.1:7860", wait_until="load"); page.wait_for_selector("#dataset-switch"); time.sleep(2)
    page.locator(".gallery-item:visible").nth(1).click(); time.sleep(1.5)
    page.locator("button:visible", has_text="Look up").click(); page.wait_for_function(JS_DONE, timeout=180000); time.sleep(2)
    import json
    print("LOOKUP", json.dumps(page.evaluate(STRUCT), indent=0)[:2500])
    page.locator("button[role=tab]:visible", has_text="Evidence").click(); time.sleep(1.5)
    r = page.evaluate(STRUCT); print("EVIDENCE lefts", json.dumps(r["lefts"])[:800]); print("EVIDENCE radii", json.dumps(r["radii"])[:800])
    b.close()
