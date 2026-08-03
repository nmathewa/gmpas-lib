"""The shared browser layout for preprocessing steps.

A deliberately reduced version of the `gmpas view` page. What a preprocessing
step needs is geography -- pan, zoom, a graticule, a scale bar and an extent
box to read a domain off -- and nothing that belongs to a model run. So the
timestep slider, the level slider, the colormap picker, the vmin/vmax boxes,
the animation panel, the exports and the probe are all absent, not hidden.

The colour scale is fixed per field rather than adjustable: a mesh field like
cell width does not change with the view, and rescaling it while panning would
make the same refinement band change colour as you move. The legend is there to
read values off; there is nothing to tune.

`page()` fills three slots so a later step (mesh generation) can reuse this
shell instead of forking it:

    title    what the tab and the sidebar heading say
    panel    extra HTML for the right-hand column, under the extent box
    script   extra JS, run after `boot()` has fetched /api/meta into `M`

A step that supplies neither `panel` nor `script` gets exactly the mesh viewer.
The server contract is four routes: `/`, `/api/meta`, `/api/frame` and
`/api/overlay`.
"""

from __future__ import annotations

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#16181c;--panel:#1e2127;--line:#2c313a;--fg:#e6e8ec;--dim:#9aa3b0;--accent:#5dcaa5}
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:250px;flex:none;background:var(--panel);border-right:1px solid var(--line);
      display:flex;flex-direction:column;overflow:hidden}
#side h1{font-size:13px;font-weight:500;margin:0;padding:12px 14px;
         border-bottom:1px solid var(--line)}
#side h1 small{display:block;color:var(--dim);font-weight:400;margin-top:2px}
.sec{padding:10px 14px;border-bottom:1px solid var(--line);flex:none}
.sec label{display:block;color:var(--dim);font-size:11px;letter-spacing:.04em;
           text-transform:uppercase;margin-bottom:6px}
input[type=text]{width:100%;background:#14161a;color:var(--fg);
  border:1px solid var(--line);border-radius:4px;padding:5px 6px;font:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
#fields{padding:2px 0}
#fields div{padding:4px 14px;margin:0 -14px;cursor:pointer;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis}
#fields div:hover{background:#252932}
#fields div.on{background:var(--accent);color:#08201a}
#facts{flex:1;overflow-y:auto}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#right{width:220px;flex:none;background:var(--panel);border-left:1px solid var(--line);
       overflow-y:auto}
#right h1{font-size:13px;font-weight:500;margin:0;padding:12px 14px;
          border-bottom:1px solid var(--line)}
#right .row{display:flex;gap:6px}
.kv{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;
    margin-bottom:4px}
.kv b{color:var(--fg);font-weight:500;font-variant-numeric:tabular-nums}
.hint{color:var(--dim);font-size:11px;margin-top:6px;line-height:1.45;
      font-variant-numeric:tabular-nums;word-break:break-word}
#top{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;
     align-items:center;color:var(--dim);flex-wrap:wrap}
#top b{color:var(--fg);font-weight:500;white-space:nowrap}
#stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;
       padding:12px;min-height:0;overflow:hidden}
#frame{display:grid;grid-template-columns:auto auto;grid-template-rows:auto auto}
#latax{position:relative;width:52px}
#lonax{position:relative;height:18px}
#corner{width:52px;height:18px}
#latax span,#lonax span{position:absolute;color:var(--dim);font-size:11px;
  font-variant-numeric:tabular-nums;white-space:nowrap}
#latax span{right:6px;transform:translateY(-50%)}
#lonax span{top:2px;transform:translateX(-50%)}
#grat{position:absolute;inset:0;pointer-events:none}
#grat i{position:absolute;background:#fff;opacity:.28}
#grat i.v{top:0;bottom:0;width:1px}
#grat i.h{left:0;right:0;height:1px}
#wrap{position:relative;line-height:0;box-shadow:0 0 0 1px var(--line);overflow:hidden;
      cursor:grab}
#wrap.drag{cursor:grabbing}
#wrap img{display:block;width:100%;height:100%;transform-origin:0 0;will-change:transform}
#over{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#scale{position:absolute;left:14px;bottom:14px;color:#fff;font-size:11px;
       text-shadow:0 0 3px #000,0 0 6px #000;pointer-events:none}
#scalebar{height:5px;border:1px solid #fff;border-top:none;box-shadow:0 0 3px #000;
          margin-top:2px}
#scaletext{display:block;font-variant-numeric:tabular-nums}
#msg{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:#000a;
     padding:4px 10px;border-radius:4px;color:var(--dim);opacity:0;transition:opacity .2s;
     pointer-events:none}
#bar{padding:8px 14px;border-top:1px solid var(--line)}
#ramp{height:14px;border-radius:2px;border:1px solid var(--line)}
#ticks{display:flex;justify-content:space-between;margin-top:3px;color:var(--dim);
       font-size:11px;font-variant-numeric:tabular-nums}
#cblabel{color:var(--dim);font-size:11px;margin-bottom:4px}
button{background:#252932;color:var(--fg);border:1px solid var(--line);border-radius:4px;
       padding:5px 9px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
</style></head><body>
<div id="side">
  <h1><span id="title">loading&hellip;</span><small id="sub"></small></h1>
  <div class="sec"><label>field</label><div id="fields"></div></div>
  <div class="sec" id="facts"><label id="factlabel">mesh</label>
    <div id="factrows"></div>
  </div>
</div>
<div id="main">
  <div id="top">
    <span>zoom</span>
    <input type="range" id="zoom" min="0" max="800" value="0" style="width:140px">
    <label style="white-space:nowrap"><input type="checkbox" id="grid" checked
      style="vertical-align:-1px"> grid</label>
    <button id="home">reset view</button>
  </div>
  <div id="stage">
    <div id="frame">
      <div id="latax"></div>
      <div id="wrap">
        <img id="data"><img id="over">
        <div id="grat"></div>
        <div id="scale"><span id="scaletext"></span><div id="scalebar"></div></div>
      </div>
      <div id="corner"></div>
      <div id="lonax"></div>
    </div>
    <div id="msg"></div>
  </div>
  <div id="bar">
    <div id="cblabel"></div>
    <div id="ramp"></div>
    <div id="ticks"></div>
  </div>
</div>
<div id="right">
  <h1>options</h1>
  <div class="sec"><label>extent</label>
    <div class="row">
      <input type="text" id="elon0" placeholder="lon min">
      <input type="text" id="elon1" placeholder="lon max"></div>
    <div class="row" style="margin-top:6px">
      <input type="text" id="elat0" placeholder="lat min">
      <input type="text" id="elat1" placeholder="lat max"></div>
    <div class="row" style="margin-top:6px">
      <button id="applyext">apply</button>
      <button id="copyext">copy</button></div>
    <div class="hint" id="exthint"></div>
  </div>
__PANEL__
</div>
<script>
const $=s=>document.querySelector(s);
let M=null, cur=null, busy=false, pend=false;
let view=null, home=null, rendered=null;      // geographic state
const ZMAX=800;                               // slider units, 100 per doubling

function say(t){const m=$("#msg");m.textContent=t;m.style.opacity=t?1:0;}
function aspect(){ return M.nx/M.ny; }

// a view is a centre plus a longitude span; latitude span follows the image
// aspect so nothing is ever stretched
function boxOf(v){
  const h=v.w/aspect();
  return [v.clon-v.w/2, v.clon+v.w/2, v.clat-h/2, v.clat+h/2];
}
function fit(box){
  const [a,b,c,d]=box, clon=(a+b)/2, clat=(c+d)/2;
  return {clon, clat, w: Math.max(b-a, (d-c)*aspect())};
}
function clamp(){
  view.w = Math.min(view.w, home.w);                 // never wider than the mesh
  view.w = Math.max(view.w, home.w/Math.pow(2,ZMAX/100));
  const h=view.w/aspect(), hh=home.w/aspect();
  const dx=Math.max(0,(home.w-view.w)/2), dy=Math.max(0,(hh-h)/2);
  view.clon=Math.min(home.clon+dx, Math.max(home.clon-dx, view.clon));
  view.clat=Math.min(home.clat+dy, Math.max(home.clat-dy, view.clat));
  $("#zoom").value = Math.round(Math.log2(home.w/view.w)*100);
}

// Frames are rendered wider than the window. Without a margin, dragging slid
// the image off its own edge and exposed blank background until the redraw
// landed. 1.4x costs ~2x the pixels and buys a screen-width of slack.
const OUTSET=1.4;
const GUTTER_X=52, GUTTER_Y=18;
function outset(b, f=OUTSET){
  const cx=(b[0]+b[1])/2, cy=(b[2]+b[3])/2, w=(b[1]-b[0])*f/2, h=(b[3]-b[2])*f/2;
  return [cx-w, cx+w, cy-h, cy+h];
}
function covers(outer, inner){
  return outer && outer[0]<=inner[0]+1e-9 && outer[1]>=inner[1]-1e-9
                && outer[2]<=inner[2]+1e-9 && outer[3]>=inner[3]-1e-9;
}
function layout(){
  if(!M) return;
  const st=$("#stage"), cs=getComputedStyle(st), pad=parseFloat(cs.padding)||0;
  const availW=st.clientWidth-2*pad-GUTTER_X, availH=st.clientHeight-2*pad-GUTTER_Y;
  const a=aspect();
  let w=availW, h=w/a;
  if(h>availH){ h=availH; w=h*a; }
  const wrap=$("#wrap");
  wrap.style.width=Math.max(80,Math.floor(w))+"px";
  wrap.style.height=Math.max(60,Math.floor(h))+"px";
}

const STEPS=[0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10,15,20,30,45,60];
function niceStep(span, want){
  for(const s of STEPS) if(span/s <= want) return s;
  return STEPS[STEPS.length-1];
}
function fmtLon(v){
  let x=((v+180)%360+360)%360-180;                 // a view can run past +180
  const r=Math.round(x*100)/100;
  if(r===0) return "0\\u00b0";
  if(Math.abs(r)===180) return "180\\u00b0";       // the antimeridian is neither
  return `${Math.abs(r)}\\u00b0${r>0?"E":"W"}`;
}
function fmtLat(v){
  const r=Math.round(v*100)/100;
  return r===0?"0\\u00b0":`${Math.abs(r)}\\u00b0${r>0?"N":"S"}`;
}
function graticule(){
  const g=$("#grat"), la=$("#latax"), lo=$("#lonax");
  g.innerHTML=""; la.innerHTML=""; lo.innerHTML="";
  if(!$("#grid").checked) return;
  const b=boxOf(view);
  const w=$("#wrap").clientWidth, h=$("#wrap").clientHeight;
  la.style.height=h+"px"; lo.style.width=w+"px";

  const sx=niceStep(b[1]-b[0], 7), sy=niceStep(b[3]-b[2], 6);
  for(let v=Math.ceil(b[0]/sx)*sx; v<=b[1]+1e-9; v+=sx){
    const f=(v-b[0])/(b[1]-b[0]);
    g.insertAdjacentHTML("beforeend",`<i class="v" style="left:${f*100}%"></i>`);
    lo.insertAdjacentHTML("beforeend",`<span style="left:${f*w}px">${fmtLon(v)}</span>`);
  }
  for(let v=Math.ceil(b[2]/sy)*sy; v<=b[3]+1e-9; v+=sy){
    const f=(b[3]-v)/(b[3]-b[2]);
    g.insertAdjacentHTML("beforeend",`<i class="h" style="top:${f*100}%"></i>`);
    la.insertAdjacentHTML("beforeend",`<span style="top:${f*h}px">${fmtLat(v)}</span>`);
  }
}
function scalebar(){
  const b=boxOf(view), wkm=(b[1]-b[0])*111.320*Math.cos(view.clat*Math.PI/180);
  const px=$("#wrap").clientWidth||600;
  let target=wkm*0.25, mag=Math.pow(10,Math.floor(Math.log10(target)));
  const nice=[1,2,5,10].map(n=>n*mag).filter(n=>n<=wkm*0.45);
  const len=nice.length?nice[nice.length-1]:target;
  $("#scalebar").style.width=(len/wkm*px)+"px";
  $("#scaletext").textContent = len>=1 ? `${len.toLocaleString()} km`
                                       : `${Math.round(len*1000).toLocaleString()} m`;
}

// The scale is a property of the field, not of the view: it is fixed once,
// from the whole mesh, so a refinement band keeps its colour while you pan.
function colorbar(){
  $("#ramp").style.background=`linear-gradient(90deg,${M.ramp.join(",")})`;
  const n=5, out=[];
  for(let i=0;i<n;i++){
    const v=cur.vmin+(cur.vmax-cur.vmin)*i/(n-1);
    out.push(`<span>${Math.abs(v)>=1e4||(v!==0&&Math.abs(v)<1e-3)?v.toExponential(2):v.toPrecision(4)}</span>`);
  }
  $("#ticks").innerHTML=out.join("");
  $("#cblabel").textContent=cur?cur.label:"";
}

async function overlay(){
  const b=outset(boxOf(view));
  $("#over").src=`/api/overlay?extent=${b.join(",")}`+
                 `&nx=${Math.round(M.nx*OUTSET)}&ny=${Math.round(M.ny*OUTSET)}`;
}
// show the frame we already have, transformed into place, until the real one
// lands. Without this a zoom looks like nothing happens and then it jumps.
function preview(){
  if(!rendered){ return; }
  const b=boxOf(view), s=(rendered[1]-rendered[0])/(b[1]-b[0]);
  const fx=(b[0]-rendered[0])/(rendered[1]-rendered[0]);
  const fy=(rendered[3]-b[3])/(rendered[3]-rendered[2]);
  const t=`translate(${-fx*s*100}%, ${-fy*s*100}%) scale(${s})`;
  $("#data").style.transform=t; $("#over").style.transform=t;
}

async function draw(){
  if(!cur) return;
  if(busy){ pend=true; return; }
  busy=true;
  // under try/finally: a fetch that rejects must not leave `busy` set, or the
  // viewer never draws again and pan/zoom look broken
  try{
    const b=outset(boxOf(view));
    const p=new URLSearchParams({field:cur.name, extent:b.join(","),
      nx:Math.round(M.nx*OUTSET), ny:Math.round(M.ny*OUTSET)});
    const t0=performance.now();
    const r=await fetch("/api/frame?"+p);
    if(!r.ok){ say((await r.json()).error); return; }
    const url=URL.createObjectURL(await r.blob());
    const img=$("#data"), old=img.src;
    img.onload=()=>{ if(old.startsWith("blob:")) URL.revokeObjectURL(old); };
    img.src=url;
    rendered=b;
    preview();        // the frame is larger than the window: crop to the view
    colorbar(); scalebar(); graticule(); panel();
    say(`${cur.label} \\u00b7 ${Math.round(performance.now()-t0)} ms`);
  }catch(e){
    say("render failed: "+e);
  }finally{
    busy=false;
    if(pend){ pend=false; draw(); }      // always drains, however we left
  }
}

function panel(){
  const b=boxOf(view);
  $("#elon0").placeholder=b[0].toFixed(2); $("#elon1").placeholder=b[1].toFixed(2);
  $("#elat0").placeholder=b[2].toFixed(2); $("#elat1").placeholder=b[3].toFixed(2);
  $("#exthint").textContent=`${b[0].toFixed(2)}, ${b[1].toFixed(2)}, `+
                            `${b[2].toFixed(2)}, ${b[3].toFixed(2)}`;
}
// The facts block is whatever the step says it is. A mesh has cells, edges and
// coverage; a distance function has none of those, and inventing values for it
// would be worse than leaving the rows out. So the server sends `subtitle` and
// `stats` (a list of [label, value] pairs) and this just renders them.
function subtitle(){
  $("#sub").textContent = M.subtitle || "";
  $("#factlabel").textContent = M.facts_label || "mesh";
  $("#factrows").innerHTML = "";
  (M.stats || []).forEach(([label, value])=>{
    const d=document.createElement("div");
    d.className="kv";
    d.innerHTML=`<span></span><b></b>`;
    d.firstChild.textContent=label;
    d.lastChild.textContent=value;
    $("#factrows").append(d);
  });
}
function fillFields(){
  $("#fields").innerHTML="";
  M.fields.forEach(f=>{
    const d=document.createElement("div");
    d.textContent=f.label; d.dataset.name=f.name; d.title=f.name;
    d.onclick=()=>pick(f.name);
    $("#fields").append(d);
  });
}
function pick(name){
  cur = M.fields.find(f=>f.name===name);
  [...$("#fields").children].forEach(d=>d.classList.toggle("on",d.dataset.name===name));
  overlay(); draw();
}

let redrawTimer=null;
function schedule(ms){ preview(); scalebar(); graticule(); clearTimeout(redrawTimer);
  redrawTimer=setTimeout(()=>{ overlay(); draw(); }, ms); }

$("#applyext").onclick = ()=>{
  const g=(id,d)=>{ const v=parseFloat($(id).value); return isFinite(v)?v:d; };
  const b=boxOf(view);
  const box=[g("#elon0",b[0]), g("#elon1",b[1]), g("#elat0",b[2]), g("#elat1",b[3])];
  if(box[1]<=box[0]||box[3]<=box[2]){ say("extent must be min then max"); return; }
  view=fit(box); clamp(); overlay(); draw(); panel();
};
$("#copyext").onclick = ()=>{
  const b=boxOf(view).map(v=>v.toFixed(4));
  navigator.clipboard?.writeText(b.join(", "));
  say("extent copied");
};
$("#home").onclick = ()=>{ view={...home}; clamp(); schedule(0); };
$("#zoom").oninput = e=>{ view.w = home.w/Math.pow(2, e.target.value/100);
                          clamp(); schedule(180); };
$("#wrap").onwheel = ev=>{
  ev.preventDefault();
  const r=$("#wrap").getBoundingClientRect();   // untransformed reference
  const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  const b=boxOf(view);
  const lon=b[0]+fx*(b[1]-b[0]), lat=b[3]-fy*(b[3]-b[2]);
  const k=Math.exp(ev.deltaY*0.0015);        // continuous, not stepped
  const nw=Math.min(home.w, Math.max(home.w/Math.pow(2,ZMAX/100), view.w*k));
  const s=nw/view.w;
  view.clon = lon + (view.clon-lon)*s;       // keep the cursor point fixed
  view.clat = lat + (view.clat-lat)*s;
  view.w = nw;
  clamp(); schedule(160);
};
let drag=null;
$("#wrap").onpointerdown = ev=>{
  drag={x:ev.clientX, y:ev.clientY, clon:view.clon, clat:view.clat, moved:false};
  try{ $("#wrap").setPointerCapture(ev.pointerId); }catch(e){}   // may refuse
  $("#wrap").classList.add("drag");
};
$("#wrap").onpointermove = ev=>{
  if(!drag) return;
  const r=$("#wrap").getBoundingClientRect(), b=boxOf(view);
  const dx=(ev.clientX-drag.x)/r.width*(b[1]-b[0]);
  const dy=(ev.clientY-drag.y)/r.height*(b[3]-b[2]);
  if(Math.abs(ev.clientX-drag.x)+Math.abs(ev.clientY-drag.y)>3) drag.moved=true;
  view.clon=drag.clon-dx; view.clat=drag.clat+dy;
  clamp(); preview(); scalebar(); graticule();
  if(!covers(rendered, boxOf(view))) schedule(90);   // ran past the margin
};
$("#wrap").onpointerup = ()=>{
  const moved=drag&&drag.moved; drag=null;
  $("#wrap").classList.remove("drag");
  if(moved) schedule(0);
};
$("#grid").onchange = graticule;
addEventListener("resize", ()=>{ layout(); scalebar(); graticule(); });

async function boot(){
  M = await (await fetch("/api/meta")).json();
  $("#title").textContent = M.file;
  home = fit(M.home); view = {...home}; rendered = null;
  layout(); subtitle(); fillFields(); panel();
  pick(M.fields[0].name);
__SCRIPT__
}
boot();
</script></body></html>
"""


def page(title: str = "gmpas prep", panel: str = "", script: str = "") -> str:
    """The preprocessing page, with the step's own controls spliced in.

    `panel` is HTML appended to the right-hand column; `script` is JS run at the
    end of `boot()`, where `M` (the /api/meta payload) is already populated.
    """
    return (_PAGE.replace("__TITLE__", title)
                 .replace("__PANEL__", panel)
                 .replace("__SCRIPT__", script))
