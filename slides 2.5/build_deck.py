"""Build the standalone deck (deck.html, speaker_notes.md) from project/deck.json and project/slides/*.html.

python3 "slides 2.5/build_deck.py"   (run from anywhere)
The same project/ files are the published Slides artifact; here the uploaded images map to img/*.png.
Arrow keys / PageUp / PageDown / space to move, Home / End, F for fullscreen, #n in the URL; print = one slide per page.
"""
import html
import json
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
deck = json.load(open("project/deck.json"))
blobs = {"/_blob/2ecd58c906eba850123237d3ba28c65d": "img/fig_task.png",
         "/_blob/d1aed2e0fda52351f7c22e6a1a8d2c11": "img/region_slide.png",
         "/_blob/f58a0c01a5de24a6dbfc01c7395fe65e": "img/survival.png",
         "/_blob/c011263d4f3366ac59f5d536d164b687": "img/evidence_weight.png",
         "/_blob/9d3fe57a6f10382b28c0f41c26609bec": "img/mech_slope.png"}
secs, notes = [], []
n = len(deck["order"])
for i, sid in enumerate(deck["order"], 1):
    s = open(f"project/slides/{sid}.html").read()
    for k, v in blobs.items():
        s = s.replace(k, v)
    m = re.search(r"<aside>(.*?)</aside>", s, re.S)
    title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S).group(1)
    notes.append(f"## {i}. {re.sub('<[^>]+>', ' ', title).strip()}\n\n{m.group(1).strip() if m else ''}\n")
    if i > 1:  # page number on every slide but the cover
        end = s.rfind("</section>")
        s = s[:end] + f'<div class="pageno">{i} / {n}</div>\n' + s[end:]
    secs.append(f'<div class="frame">{s}</div>')
fonts = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values())
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deck["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>{fonts}
<style>
*{{box-sizing:border-box}} html,body{{margin:0;background:#0d1426}}
section{{width:1920px;height:1080px;overflow:hidden;position:relative}}
section h1,section h2,section h3,section p,section ul,section ol{{margin:0}}
section aside{{display:none}}
section table{{border-collapse:collapse;width:100%}}
section th,section td{{padding:.45em .6em;text-align:left;vertical-align:top;border-bottom:1px solid #D9D4C7}}
section th{{font-weight:600;border-bottom:2px solid #14213D}}
img{{max-width:100%}}
.pageno{{position:absolute;right:64px;bottom:44px;font-size:24px;color:#8A93A6;font-family:'IBM Plex Sans',Arial,sans-serif;font-variant-numeric:tabular-nums}}
@media screen{{
  html,body{{height:100%;overflow:hidden}}
  .frame{{position:absolute;left:50%;top:50%;width:1920px;height:1080px;transform:translate(-50%,-50%) scale(var(--s,0.5));display:none}}
  .frame.on{{display:block}}
}}
@media print{{
  @page{{size:1920px 1080px;margin:0}}
  .frame{{width:1920px;height:1080px;page-break-after:always;break-after:page}}
}}
</style></head><body>
{chr(10).join(secs)}
<script>
var frames=[].slice.call(document.querySelectorAll('.frame')),cur=0;
function fit(){{document.documentElement.style.setProperty('--s',Math.min(innerWidth/1920,innerHeight/1080));}}
function show(n){{cur=Math.max(0,Math.min(frames.length-1,n));
  frames.forEach(function(f,k){{f.classList.toggle('on',k===cur);}});
  try{{history.replaceState(null,'','#'+(cur+1));}}catch(e){{}}}}
addEventListener('resize',fit);
addEventListener('keydown',function(e){{
  if(e.metaKey||e.ctrlKey||e.altKey)return;var k=e.key;
  if(k==='ArrowRight'||k==='ArrowDown'||k==='PageDown'||(k===' '&&!e.shiftKey)){{e.preventDefault();show(cur+1);}}
  else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'||(k===' '&&e.shiftKey)){{e.preventDefault();show(cur-1);}}
  else if(k==='Home'){{e.preventDefault();show(0);}}
  else if(k==='End'){{e.preventDefault();show(frames.length-1);}}
  else if(k==='f'||k==='F'){{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen();}}
}});
var tx=null;
addEventListener('touchstart',function(e){{tx=e.changedTouches[0].clientX;}},{{passive:true}});
addEventListener('touchend',function(e){{if(tx===null)return;var dx=e.changedTouches[0].clientX-tx;tx=null;
  if(Math.abs(dx)>40)show(cur+(dx<0?1:-1));}},{{passive:true}});
addEventListener('hashchange',function(){{var n=parseInt(location.hash.slice(1),10);if(n&&n-1!==cur)show(n-1);}});
fit();show((parseInt(location.hash.slice(1),10)||1)-1);
</script>
</body></html>"""
open("deck.html", "w").write(page)
open("speaker_notes.md", "w").write(f"# {deck['title']}: speaker notes\n\n" + "\n".join(notes))
print(f"deck.html: {n} slides")
