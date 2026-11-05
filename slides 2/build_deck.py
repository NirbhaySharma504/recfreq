"""Build the standalone deck (deck.html, speaker_notes.md) from source/deck.json and source/slides/*.html.

python3 slides/build_deck.py   (run from the repo root or from slides/)
The deck shows one slide at a time (arrow keys, PageUp/PageDown, space, Home/End, F for fullscreen, swipe;
#n in the URL); every slide but the cover gets a page number. Printing still gives one slide per page.
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
    if i > 1:  # page number on every slide but the cover
        end = s.rfind("</section>")
        s = s[:end] + f'<div class="pageno">{i} / {len(deck["order"])}</div>\n' + s[end:]
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
.pageno{{position:absolute;right:64px;bottom:44px;font-size:24px;color:#6B7487;font-variant-numeric:tabular-nums}}
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
// One slide at a time: arrows / PageUp / PageDown / space to move, Home / End, F for fullscreen; #n in the URL
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
open("speaker_notes.md", "w").write(f"# {deck['title']} — speaker notes\n\n" + "\n".join(notes))
print(len(secs), "slides -> deck.html, speaker_notes.md")
