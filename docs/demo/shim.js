// Run the gmpas viewer page with no Python behind it.
//
// The page talks to its server through relative `fetch("api/...")` calls and
// one `<img src="api/overlay">`. This intercepts both and answers them from
// files baked beside the page by bake.py, so the published demo is the real
// interface rather than a picture of it.
//
// The only part that is a reimplementation -- and so the only part that can
// drift from the Python -- is the rasterizer below. It reproduces ViewIndex
// in src/gmpas/viewer.py: nearest cell centre on the unit sphere, blanked
// where the pixel is further away than twice that cell's own radius. The
// test suite renders the same frames both ways and compares the pixels.

(() => {
  "use strict";

  const D = "data/";
  const state = {meta: null, fields: null, ramps: null, coast: null,
                 xyz: null, radius: null, lon: null, lat: null, cells: 0,
                 buckets: null, cache: new Map()};

  const getJSON = p => fetch(D + p).then(r => r.json());

  async function load() {
    const [meta, fields, coast, info] = await Promise.all(
      ["meta.json", "fields.json", "coast.json", "mesh.json"].map(getJSON));
    Object.assign(state, {meta, fields, coast, cells: info.cells});

    const buf = await (await fetch(D + "mesh.bin")).arrayBuffer();
    const n = info.cells;
    state.lon = new Float32Array(buf, 0, n);
    state.lat = new Float32Array(buf, n * 4, n);
    state.radius = new Float32Array(buf, n * 8, n);

    // Unit vectors, rebuilt here rather than shipped: 197 KB saved against a
    // rounding difference of 8.6e-08, where the smallest cell radius the
    // comparison has to resolve is 3.5e-03.
    const R = Math.PI / 180, xyz = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      const la = state.lat[i] * R, lo = state.lon[i] * R, c = Math.cos(la);
      xyz[i * 3] = c * Math.cos(lo);
      xyz[i * 3 + 1] = c * Math.sin(lo);
      xyz[i * 3 + 2] = Math.sin(la);
    }
    state.xyz = xyz;
    buckets();
  }

  // Colormaps are a file each, fetched when chosen: all 63 at 256 stops is
  // 174 KB of first paint spent on colormaps nobody has picked.
  const ramps = new Map();
  async function rampOf(name) {
    const want = state.ramps && state.ramps[name] ? name
               : (name && state.meta.cmaps.includes(name) ? name : "viridis");
    if (!ramps.has(want)) {
      try {
        ramps.set(want, (await getJSON(`ramps/${want.replace("/", "_")}.json`))
                          .map(hex));
      } catch (e) {
        ramps.set(want, (state.meta.ramps[want] || []).map(hex));
      }
    }
    return ramps.get(want);
  }

  // -- nearest cell -------------------------------------------------------
  // Brute force is 1.6M pixels x 8k cells a frame, far too slow, so cells go
  // into a lon/lat bucket grid and a pixel searches its own bucket then rings
  // outward.
  //
  // The grid covers the *mesh's* bounding box, not the globe. A regional mesh
  // like this one occupies about 3% of the sphere, so a global grid left 97%
  // of its buckets empty and crammed ~55 cells into each of the rest; sized
  // to the mesh instead, each bucket holds two or three. Pixels outside the
  // box are off the mesh by definition and cost nothing at all -- which is
  // most of them, the moment anyone zooms out.
  const B = {x0: 0, y0: 0, dx: 1, dy: 1, nx: 1, ny: 1, pad: 1.0};

  function buckets() {
    const {lon, lat, cells} = state;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < cells; i++) {
      if (lon[i] < x0) x0 = lon[i];
      if (lon[i] > x1) x1 = lon[i];
      if (lat[i] < y0) y0 = lat[i];
      if (lat[i] > y1) y1 = lat[i];
    }
    // aim for ~2 cells a bucket, and keep them roughly square in degrees
    const side = Math.sqrt((x1 - x0) * (y1 - y0) / Math.max(1, cells / 2));
    B.nx = Math.max(1, Math.ceil((x1 - x0) / side));
    B.ny = Math.max(1, Math.ceil((y1 - y0) / side));
    B.x0 = x0; B.y0 = y0;
    B.dx = (x1 - x0) / B.nx || 1; B.dy = (y1 - y0) / B.ny || 1;
    B.x1 = x1; B.y1 = y1;

    const counts = new Int32Array(B.nx * B.ny);
    const at = new Int32Array(cells);
    for (let i = 0; i < cells; i++) {
      at[i] = cellBucket(lon[i], lat[i]);
      counts[at[i]]++;
    }
    const start = new Int32Array(B.nx * B.ny + 1);
    for (let k = 0; k < counts.length; k++) start[k + 1] = start[k] + counts[k];
    const items = new Int32Array(cells), fill = start.slice(0, -1);
    for (let i = 0; i < cells; i++) items[fill[at[i]]++] = i;
    state.buckets = {start, items};
  }

  function cellBucket(lonDeg, latDeg) {
    const bx = Math.min(B.nx - 1, Math.max(0, Math.floor((lonDeg - B.x0) / B.dx)));
    const by = Math.min(B.ny - 1, Math.max(0, Math.floor((latDeg - B.y0) / B.dy)));
    return by * B.nx + bx;
  }

  function nearest(px, py, pz, lonDeg, latDeg) {
    // outside the mesh's box by more than any cell could reach
    if (lonDeg < B.x0 - B.pad || lonDeg > B.x1 + B.pad ||
        latDeg < B.y0 - B.pad || latDeg > B.y1 + B.pad) return [-1, Infinity];
    const bx = Math.min(B.nx - 1, Math.max(0, Math.floor((lonDeg - B.x0) / B.dx)));
    const by = Math.min(B.ny - 1, Math.max(0, Math.floor((latDeg - B.y0) / B.dy)));
    const {start, items} = state.buckets, xyz = state.xyz;
    const R = Math.PI / 180, step = Math.min(B.dx, B.dy) * R;
    let best = -1, bestD = Infinity;
    const far = Math.max(B.nx, B.ny);
    for (let ring = 0; ring < far; ring++) {
      // nothing in a further ring can beat this, so stop
      if (best >= 0 && ring > 1 && bestD < ((ring - 1) * step) ** 2) break;
      const ylo = Math.max(0, by - ring), yhi = Math.min(B.ny - 1, by + ring);
      for (let yy = ylo; yy <= yhi; yy++) {
        const edgeRow = Math.abs(yy - by) === ring;
        const xlo = Math.max(0, bx - ring), xhi = Math.min(B.nx - 1, bx + ring);
        for (let xx = xlo; xx <= xhi; xx++) {
          if (!edgeRow && Math.abs(xx - bx) !== ring) continue;
          const b = yy * B.nx + xx;
          for (let k = start[b]; k < start[b + 1]; k++) {
            const i = items[k], j = i * 3;
            const ex = xyz[j] - px, ey = xyz[j + 1] - py, ez = xyz[j + 2] - pz;
            const d = ex * ex + ey * ey + ez * ez;
            if (d < bestD) { bestD = d; best = i; }
          }
        }
      }
    }
    return [best, Math.sqrt(bestD)];
  }

  // -- the view index, as ViewIndex builds it -----------------------------
  function viewIndex(extent, nx, ny) {
    const key = extent.map(v => v.toFixed(6)).join(",") + `:${nx}x${ny}`;
    const hit = state.cache.get(key);
    if (hit) return hit;
    const [lon0, lon1, lat0, lat1] = extent;
    const idx = new Int32Array(nx * ny), blank = new Uint8Array(nx * ny);
    const R = Math.PI / 180;
    for (let y = 0; y < ny; y++) {
      // pixel centres, matching raster.grid_points
      const latDeg = lat1 - (y + 0.5) * (lat1 - lat0) / ny;
      const la = latDeg * R, cla = Math.cos(la), sla = Math.sin(la);
      for (let x = 0; x < nx; x++) {
        const lonDeg = lon0 + (x + 0.5) * (lon1 - lon0) / nx;
        const lo = lonDeg * R;
        const [cell, dist] = nearest(cla * Math.cos(lo), cla * Math.sin(lo), sla,
                                     lonDeg, latDeg);
        const p = y * nx + x;
        if (cell < 0 || dist > 2 * state.radius[cell]) { blank[p] = 1; idx[p] = 0; }
        else idx[p] = cell;
      }
    }
    const out = {idx, blank, nx, ny};
    if (state.cache.size > 6) state.cache.delete(state.cache.keys().next().value);
    state.cache.set(key, out);
    return out;
  }

  // -- values -------------------------------------------------------------
  const values = new Map();
  async function field(name, level) {
    if (!values.has(name)) {
      const buf = await (await fetch(`${D}f/${encodeURIComponent(name)}.bin`))
        .arrayBuffer();
      values.set(name, new Float32Array(buf));
    }
    const all = values.get(name);
    return all.subarray(level * state.cells, (level + 1) * state.cells);
  }

  const hex = s => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16),
                    parseInt(s.slice(5, 7), 16)];

  async function framePNG(q) {
    const extent = q.get("extent").split(",").map(Number);
    const nx = +q.get("nx") || 1200, ny = +q.get("ny") || 700;
    const name = q.get("var"), level = +(q.get("level") || 0);
    const vals = await field(name, level);
    const view = viewIndex(extent, nx, ny);

    let lo = q.get("vmin") ? +q.get("vmin") : state.fields[name].lo;
    let hi = q.get("vmax") ? +q.get("vmax") : state.fields[name].hi;
    if (!(hi > lo)) hi = lo + 1;

    const stops = await rampOf(q.get("cmap"));
    const last = stops.length - 1;
    const img = new ImageData(nx, ny);
    const px = img.data;
    for (let p = 0; p < nx * ny; p++) {
      const o = p * 4;
      if (view.blank[p]) { px[o + 3] = 0; continue; }
      const v = vals[view.idx[p]];
      if (!Number.isFinite(v)) { px[o + 3] = 0; continue; }
      const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
      const c = stops[Math.round(t * last)];
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
    const cv = new OffscreenCanvas(nx, ny);
    cv.getContext("2d").putImageData(img, 0, 0);
    const blob = await cv.convertToBlob({type: "image/png"});
    // X-Range is not optional: the page does headers.get("X-Range").split(),
    // so a missing one is a TypeError and a blank map.
    return new Response(blob, {status: 200, headers: {
      "Content-Type": "image/png", "X-Range": `${lo},${hi}`}});
  }

  function probe(q) {
    const lonDeg = +q.get("lon"), latDeg = +q.get("lat"), R = Math.PI / 180;
    const la = latDeg * R, lo = lonDeg * R, cla = Math.cos(la);
    const [cell, dist] = nearest(cla * Math.cos(lo), cla * Math.sin(lo),
                                 Math.sin(la), lonDeg, latDeg);
    const off = cell < 0 || dist > 2 * state.radius[cell];
    const vals = values.get(q.get("var"));
    const level = +(q.get("level") || 0);
    let v = null;
    if (!off && vals) {
      const raw = vals[level * state.cells + cell];
      v = Number.isFinite(raw) ? raw : null;     // never NaN: JSON.parse rejects it
    }
    return json(off ? {cell: -1, lon: lonDeg, lat: latDeg, value: null}
                    : {cell, lon: +state.lon[cell].toFixed(4),
                       lat: +state.lat[cell].toFixed(4), value: v});
  }

  const json = (o, status = 200) => new Response(JSON.stringify(o), {
    status, headers: {"Content-Type": "application/json"}});

  // -- coastlines, in place of the server's PNG ---------------------------
  window.GMPAS_OVERLAY = extent => {
    const el = document.querySelector("#over");
    if (!el || !state.coast) return;
    const [lon0, lon1, lat0, lat1] = extent;
    const w = el.clientWidth || 1200, h = el.clientHeight || 700;
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const g = cv.getContext("2d");
    g.strokeStyle = "#111"; g.lineWidth = 0.8; g.beginPath();
    for (const line of state.coast) {
      for (let i = 0; i < line.length; i++) {
        const x = (line[i][0] - lon0) / (lon1 - lon0) * w;
        const y = (lat1 - line[i][1]) / (lat1 - lat0) * h;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
    }
    g.stroke();
    el.src = cv.toDataURL("image/png");
  };

  // -- interception -------------------------------------------------------
  const real = window.fetch.bind(window);
  const ready = load();

  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    const at = url.indexOf("api/");
    if (at < 0 || url.startsWith(D)) return real(input, init);
    await ready;
    const path = url.slice(at).split("?")[0];
    const q = new URLSearchParams(url.split("?")[1] || "");
    try {
      if (path === "api/meta") return json(state.meta);
      if (path === "api/status") return json({scanning: false,
        steps: state.meta.steps, labels: state.meta.labels});
      if (path === "api/frame") return await framePNG(q);
      if (path === "api/probe") return probe(q);
      if (path === "api/series") {
        if (q.get("warm") || q.get("stop")) return json({state: "ok"});
        const p = probe(q), d = await p.json();
        return json({state: "done", cell: d.cell, lon: d.lon, lat: d.lat,
                     label: state.meta.variables.find(v => v.name === q.get("var"))
                            ?.label || q.get("var"),
                     labels: state.meta.labels, values: [d.value]});
      }
    } catch (e) {
      return json({error: `${e.name}: ${e.message}`}, 500);
    }
    // Everything left needs matplotlib on a server. Say so in the shape the
    // page expects -- several call sites do (await r.json()).error, and a
    // non-JSON body turns a tidy message into an unhandled rejection.
    return json({error: "this is a static demo: figures, animations and " +
                        "exports need gmpas running locally"}, 500);
  };
})();
