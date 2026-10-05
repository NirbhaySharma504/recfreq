"""Build the standalone deck (deck.html, speaker_notes.md) from source/deck.json and source/slides/*.html.

python3 slides/build_deck.py   (run from the repo root or from slides/)
Images: the published deck references uploaded blobs; here they map to img/*.png.
"""
import html
import json
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
deck = json.load(open("source/deck.json"))
blobs = {"/_blob/7a07ec398f36e012629697bf725e8910": "img/fig_task.png",
         "/_blob/ff65a29fa16f26dd12b2c3c53560f8b0": "img/pe_control_nw3.png",
         "/_blob/064138d8c4b843a42c451fa939e9ec41": "img/effective_params.png",
         "/_blob/6bdb38b3a2dd3809050fe1653798feeb": "img/fig_slope.png",
         "/_blob/c68cc4eafd1341b3eaf1ceba440193c6": "img/mechanism.png"}
extra = json.load(open("source/blobs.json")) if os.path.exists("source/blobs.json") else {}
blobs.update(extra)
secs, notes = [], []
for i, sid in enumerate(deck["order"], 1):
    s = open(f"source/slides/{sid}.html").read()
    for k, v in blobs.items():
        s = s.replace(k, v)
    m = re.search(r"<aside>(.*?)</aside>", s, re.S)
    title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S).group(1)
    notes.append(f"## {i}. {re.sub('<[^>]+>', '', title)}\n\n{m.group(1).strip() if m else ''}\n")
    secs.append(f'<div class="frame">{s}</div>')
fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values())
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deck["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>{fonts}
<style>
*{{box-sizing:border-box}} html,body{{margin:0;background:#0d1426}}
section{{width:1920px;height:1080px;overflow:hidden;position:relative}}
section aside{{display:none}}
img{{max-width:100%}}
@media screen{{
  body{{padding:16px 0}}
  .frame{{width:calc(100vw - 32px);aspect-ratio:16/9;margin:0 auto 16px;overflow:hidden;position:relative;box-shadow:0 4px 24px rgba(0,0,0,.4)}}
  .frame>section{{transform-origin:0 0;transform:scale(var(--s,0.5));position:absolute;top:0;left:0}}
}}
@media print{{
  @page{{size:1920px 1080px;margin:0}}
  .frame{{width:1920px;height:1080px;page-break-after:always;break-after:page}}
}}
</style></head><body>
{chr(10).join(secs)}
<script>
function fit(){{var f=document.querySelector('.frame');if(f)document.documentElement.style.setProperty('--s',f.clientWidth/1920);}}
addEventListener('resize',fit);fit();
</script>
</body></html>"""
open("deck.html", "w").write(page)
open("speaker_notes.md", "w").write(f"# {deck['title']} — speaker notes\n\n" + "\n".join(notes))
print(len(secs), "slides -> deck.html, speaker_notes.md")
