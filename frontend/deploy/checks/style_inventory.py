import json, time, sys
from playwright.sync_api import sync_playwright
JS_DONE = """() => { const v=[...document.querySelectorAll('#lk-verdict')].find(e=>e.offsetParent); return v && (v.innerText.includes('confidence') || v.innerText.includes('Unfamiliar')); }"""
INVENTORY = r"""() => {
  const lum = (c) => { const m = c.match(/[\d.]+/g); if (!m) return null; let [r,g,b] = m.slice(0,3).map(Number);
     if (c.startsWith('oklch')) return null; const f=v=>{v/=255; return v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055,2.4)};
     return 0.2126*f(r)+0.7152*f(g)+0.0722*f(b); };
  const bgOf = (e) => { for (let x=e; x; x=x.parentElement) { const b=getComputedStyle(x).backgroundColor; if (b && !b.includes('0, 0, 0, 0') && b!=='transparent') return b; } return 'rgb(255,255,255)'; };
  const rows = [];
  for (const e of document.querySelectorAll('body *')) {
    if (!e.offsetParent || !e.childNodes.length) continue;
    const own = [...e.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim()).map(n=>n.textContent.trim()).join(' ');
    if (!own) continue;
    const s = getComputedStyle(e);
    rows.push({tag: e.tagName.toLowerCase(), cls: String(e.className).split(' ').filter(c=>!c.startsWith('svelte')).slice(0,2).join('.'),
               size: s.fontSize, weight: s.fontWeight, color: s.color, family: s.fontFamily.split(',')[0], bg: bgOf(e), text: own.slice(0,40)});
  }
  return rows; }"""
with sync_playwright() as p:
    b = p.chromium.launch(); page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.goto("http://127.0.0.1:7860", wait_until="load"); page.wait_for_selector("#dataset-switch"); time.sleep(2)
    out = {}
    out["bach_slides"] = page.evaluate(INVENTORY)
    page.locator("button[role=tab]:visible", has_text="Lookup").click(); time.sleep(1.5)  # Slide view opens first
    page.locator(".gallery-item:visible").nth(1).click(); time.sleep(1.5)
    page.locator("button:visible", has_text="Look up").click(); page.wait_for_function(JS_DONE, timeout=180000); time.sleep(1.5)
    page.locator(".refs-fold > summary:visible").click(); page.locator(".label-wrap:visible").last.click(); time.sleep(1)
    out["bach_lookup"] = page.evaluate(INVENTORY)
    page.locator("button[role=tab]:visible", has_text="Evidence").click(); time.sleep(1.5)
    out["bach_evidence"] = page.evaluate(INVENTORY)
    page.locator("#dataset-switch label", has_text="BreakHis").click(); time.sleep(1.5)
    page.locator("button[role=tab]:visible", has_text="Evidence").click(); time.sleep(1.5)
    out["breakhis_evidence"] = page.evaluate(INVENTORY)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    b.close()
