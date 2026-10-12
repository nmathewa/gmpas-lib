// Run the gmpas viewer page with no Python behind it.
//
// The page talks to its server through relative `fetch("api/...")` calls and
// one `<img src="api/overlay">`. This intercepts both and answers them from
// files baked beside the page by bake.py, so the published demo is the real
// interface rather than a picture of it.
//
// Three parts are reimplementations, and so can drift from the Python:
//
// - the rasterizer: ViewIndex in src/gmpas/viewer.py, nearest cell centre on
//   the unit sphere, blanked where the pixel is further away than twice that
//   cell's own radius;
// - the map projections: the inverses gmpas/projection.py gets from cartopy,
//   for the ones with a short closed form;
// - the colour options: gmpas/colour.py and gmpas/palettes, levels included.
//
// The test suite renders the same frames both ways and compares the pixels.

(() => {
  "use strict";

  const D = "data/";
  const state = {meta: null, fields: null, ramps: null, coast: null,
                 xyz: null, radius: null, lon: null, lat: null, cells: 0,
                 buckets: null, cache: new Map(), proj: {}, cmapN: {},
                 world: null, lastOverlay: null};

  const getJSON = p => fetch(D + p).then(r => r.json());

  async function load() {
    const [meta, fields, coast, info] = await Promise.all(
      ["meta.json", "fields.json", "coast.json", "mesh.json"].map(getJSON));
    Object.assign(state, {meta, fields, coast, cells: info.cells,
                          proj: info.proj || {}, cmapN: info.cmap_n || {}});
    // The world's coastlines, for a projected map that shows more than the
    // region: fetched after first paint, and the coast redrawn when it lands.
    getJSON("coast_world.json").then(w => {
      state.world = w;
      if (state.lastOverlay) window.GMPAS_OVERLAY(...state.lastOverlay);
    }).catch(() => {});

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
  function viewIndex(extent, nx, ny, pj) {
    const key = extent.map(v => v.toFixed(6)).join(",") + `:${nx}x${ny}` +
                (pj ? ":" + pj.key : "");
    const hit = state.cache.get(key);
    if (hit) return hit;
    const [lon0, lon1, lat0, lat1] = extent;
    const idx = new Int32Array(nx * ny), blank = new Uint8Array(nx * ny);
    const R = Math.PI / 180;
    for (let y = 0; pj && y < ny; y++) {
      // a projected box is metres: invert each pixel centre to lon/lat, as
      // projection.projected_points does, then search as below
      const Y = lat1 - (y + 0.5) * (lat1 - lat0) / ny;
      for (let x = 0; x < nx; x++) {
        const p = y * nx + x;
        const ll = pj.inv(lon0 + (x + 0.5) * (lon1 - lon0) / nx, Y);
        if (!ll) { blank[p] = 1; continue; }
        const la = ll[1] * R, lo = ll[0] * R, cla = Math.cos(la);
        const [cell, dist] = nearest(cla * Math.cos(lo), cla * Math.sin(lo),
                                     Math.sin(la), ll[0], ll[1]);
        if (cell < 0 || dist > 2 * state.radius[cell]) blank[p] = 1;
        else idx[p] = cell;
      }
    }
    for (let y = 0; !pj && y < ny; y++) {
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

  // -- map projections, as gmpas/projection.py has them from cartopy -------
  // The same maps cartopy builds: orthographic on a sphere of the WGS84
  // radius, the others on the WGS84 ellipsoid (Snyder's formulas, as PROJ
  // codes them). `inv` answers null off the map, where projection.unproject
  // answers on=False.
  const PA = 6378137, PF = 1 / 298.257223563, PE2 = PF * (2 - PF),
        PE = Math.sqrt(PE2), RAD = Math.PI / 180;
  const tsfn = phi => {
    const es = PE * Math.sin(phi);
    return Math.tan(Math.PI / 4 - phi / 2) / Math.pow((1 - es) / (1 + es), PE / 2);
  };
  const msfn = phi => {
    const sn = Math.sin(phi);
    return Math.cos(phi) / Math.sqrt(1 - PE2 * sn * sn);
  };
  const phi2 = ts => {
    let phi = Math.PI / 2 - 2 * Math.atan(ts);
    for (let i = 0; i < 15; i++) {
      const es = PE * Math.sin(phi);
      const next = Math.PI / 2 - 2 * Math.atan(ts * Math.pow((1 - es) / (1 + es), PE / 2));
      const done = Math.abs(next - phi) < 1e-13;
      phi = next;
      if (done) break;
    }
    return phi;
  };
  const wrap180 = d => ((d + 180) % 360 + 360) % 360 - 180;

  function makeProj(name, plon, plat, psp) {
    const lon0 = +plon || 0, key = [name, plon, plat, psp].join("|");
    const dl = lon => wrap180(lon - lon0) * RAD;
    const out = (lam, phi) => [wrap180(lon0 + lam / RAD), phi / RAD];
    if (name === "Orthographic") {
      const phi0 = (+plat || 0) * RAD, s0 = Math.sin(phi0), c0 = Math.cos(phi0);
      return {key,
        fwd(lon, lat) {
          const l = dl(lon), phi = lat * RAD, cp = Math.cos(phi), sp = Math.sin(phi);
          if (s0 * sp + c0 * cp * Math.cos(l) < 0) return null;   // far side
          return [PA * cp * Math.sin(l), PA * (c0 * sp - s0 * cp * Math.cos(l))];
        },
        inv(x, y) {
          const rho = Math.hypot(x, y);
          if (rho > PA) return null;
          const c = Math.asin(rho / PA), sc = Math.sin(c), cc = Math.cos(c);
          const phi = Math.asin(Math.max(-1, Math.min(1,
            cc * s0 + (rho > 0 ? y * sc * c0 / rho : 0))));
          return out(Math.atan2(x * sc, rho * cc * c0 - y * sc * s0), phi);
        }};
    }
    if (name === "Mercator") {
      return {key,
        fwd(lon, lat) {
          if (Math.abs(lat) >= 90) return null;
          return [PA * dl(lon), -PA * Math.log(tsfn(lat * RAD))];
        },
        inv(x, y) {
          if (Math.abs(x) > PA * Math.PI) return null;
          return out(x / PA, phi2(Math.exp(-y / PA)));
        }};
    }
    if (name === "NorthPolarStereo" || name === "SouthPolarStereo") {
      const sgn = name === "NorthPolarStereo" ? 1 : -1;
      const k = 2 * PA / Math.sqrt(Math.pow(1 + PE, 1 + PE) * Math.pow(1 - PE, 1 - PE));
      return {key,
        fwd(lon, lat) {
          if (sgn * lat <= -90) return null;                      // the far pole
          const rho = k * tsfn(sgn * lat * RAD), l = dl(lon);
          return [rho * Math.sin(l), -sgn * rho * Math.cos(l)];
        },
        inv(x, y) {
          return out(Math.atan2(x, -sgn * y), sgn * phi2(Math.hypot(x, y) / k));
        }};
    }
    if (name === "LambertConformal") {
      const phi0 = (plat == null ? 40 : +plat) * RAD;
      const sp = psp ? String(psp).split(",").map(Number)
                     : (phi0 < 0 ? [-33, -45] : [33, 45]);
      const p1 = sp[0] * RAD, p2 = sp[1] * RAD;
      const m1 = msfn(p1), t1 = tsfn(p1);
      const n = Math.abs(p1 - p2) < 1e-10 ? Math.sin(p1)
              : Math.log(m1 / msfn(p2)) / Math.log(t1 / tsfn(p2));
      const c = m1 * Math.pow(t1, -n) / n;
      const rho0 = Math.abs(Math.abs(phi0) - Math.PI / 2) < 1e-10 ? 0
                 : c * Math.pow(tsfn(phi0), n);
      return {key,
        fwd(lon, lat) {
          if (n * lat <= -90 * Math.abs(n)) return null;          // the far pole
          const rho = c * Math.pow(tsfn(lat * RAD), n), th = n * dl(lon);
          return [PA * rho * Math.sin(th), PA * (rho0 - rho * Math.cos(th))];
        },
        inv(x, y) {
          let X = x / PA, Y = rho0 - y / PA, rho = Math.hypot(X, Y);
          if (n < 0) { rho = -rho; X = -X; Y = -Y; }
          const lam = Math.atan2(X, Y) / n;
          if (Math.abs(lam) > Math.PI) return null;               // past the cone's cut
          return out(lam, rho === 0 ? Math.sign(n) * Math.PI / 2
                                    : phi2(Math.pow(rho / c, 1 / n)));
        }};
    }
    throw new Error(`the demo has no ${name} map`);
  }

  const projCache = new Map();
  function projOf(q) {
    if (!q.get("proj")) return null;
    const args = [q.get("proj"), q.get("plon"), q.get("plat"), q.get("psp")];
    const key = args.join("|");
    if (!projCache.has(key)) projCache.set(key, makeProj(...args));
    return projCache.get(key);
  }

  // -- colour options, as gmpas/colour.py and gmpas/palettes have them ------
  const MAX_STEPS = 252;
  const f64 = new Float64Array(1), i64 = new BigInt64Array(f64.buffer);
  function nextAfter(v, up) {
    if (v === 0) return up ? Number.MIN_VALUE : -Number.MIN_VALUE;
    f64[0] = v;
    i64[0] += ((v > 0) === up) ? 1n : -1n;
    return f64[0];
  }
  const g6 = v => String(+(+v).toPrecision(6));
  const NUM = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?";
  const isNum = t => new RegExp(`^${NUM}$`).test(t);
  const FORMS = "N (N equal steps), (lo,hi,step) segments such as " +
    "(-25,0,5)(0,25,1), (,,step) over the colour range, single levels " +
    "such as (0)(1)(2)(5), and (-inf) / (inf) for open ends";

  function parseLevels(text) {
    const t = String(text).replace(/\s+/g, "").toLowerCase();
    if (!t) throw new Error("levels is empty");
    if (/^\d+$/.test(t)) {
      const n = parseInt(t, 10);
      if (!(n >= 1 && n <= MAX_STEPS))
        throw new Error(`levels ${n}: between 1 and ${MAX_STEPS} steps fit`);
      return {count: n};
    }
    const groups = [...t.matchAll(/\(([^()]*)\)/g)].map(m => m[1]);
    if (!groups.length || groups.map(g => `(${g})`).join("") !== t)
      throw new Error(`levels '${text}' is not understood; use ${FORMS}`);
    const segments = []; let low = false, high = false;
    groups.forEach((g, i) => {
      const parts = g.split(",");
      if (parts.length === 1 && /^[-+]?inf$/.test(parts[0])) {
        const neg = parts[0].startsWith("-");
        if (neg && i === 0) low = true;
        else if (!neg && i === groups.length - 1) high = true;
        else throw new Error(`levels: (${g}) belongs at the ${neg ? "start" : "end"}`);
        return;
      }
      if (parts.length === 1 && isNum(parts[0])) {
        segments.push(["value", parseFloat(parts[0])]);
        return;
      }
      if (parts.length === 3 && isNum(parts[2]) &&
          parts.slice(0, 2).every(p => p === "" || isNum(p))) {
        const [lo, hi, step] = parts.map(p => p ? parseFloat(p) : null);
        if (!(step > 0)) throw new Error(`levels: (${g}) needs a step above zero`);
        if (lo !== null && hi !== null && !(hi > lo))
          throw new Error(`levels: (${g}) needs hi above lo`);
        segments.push(["range", lo, hi, step]);
        return;
      }
      throw new Error(`levels: (${g}) is not understood; use ${FORMS}`);
    });
    if (!segments.length) throw new Error("levels: open ends need levels between them");
    return {segments, open: [low, high]};
  }

  function levelsEnds(parsed) {
    if ("count" in parsed) return [null, null];
    const first = parsed.segments[0], last = parsed.segments.at(-1);
    return [first[1], last[0] === "value" ? last[1] : last[2]];
  }

  // Edges as the Python makes them, but with the top one left on its value:
  // `nudged` is the same list with the top edge just past it, so a field's
  // maximum on the last level falls in the last step.
  function bandEdges(lo, hi, bands) {
    // numpy.linspace: start + i*step, the last one the stop itself
    const step = (hi - lo) / bands;
    return Array.from({length: bands + 1}, (_, i) => i === bands ? hi : lo + i * step);
  }
  function intervalEdges(lo, hi, step) {
    if (!(step > 0)) throw new Error(`a colour step of ${g6(step)} must be above zero`);
    const i0 = Math.floor(lo / step + 1e-9);
    let i1 = Math.ceil(hi / step - 1e-9);
    if (i1 <= i0) i1 = i0 + 1;
    if (i1 - i0 > MAX_STEPS)
      throw new Error(`a colour step of ${g6(step)} over ${g6(lo)}..${g6(hi)} makes ` +
        `${i1 - i0} steps; at most ${MAX_STEPS} fit -- use a larger step or a narrower range`);
    return Array.from({length: i1 - i0 + 1}, (_, i) => (i0 + i) * step);
  }
  function levelEdges(parsed, lo, hi) {
    if ("count" in parsed) return bandEdges(lo, hi, parsed.count);
    const out = [];
    for (const seg of parsed.segments) {
      let part;
      if (seg[0] === "value") part = [seg[1]];
      else {
        let [, a, b, step] = seg;
        if (a === null && b === null) part = intervalEdges(lo, hi, step);
        else {
          a = a === null ? lo : a; b = b === null ? hi : b;
          if (!(b > a)) throw new Error(
            `levels: the segment runs from ${g6(a)} to ${g6(b)}, which is empty`);
          const n = Math.floor((b - a) / step + 1e-9);
          if (n > MAX_STEPS) throw new Error(`levels: ${g6(a)} to ${g6(b)} every ` +
            `${g6(step)} makes ${n} steps; at most ${MAX_STEPS} fit`);
          part = Array.from({length: n + 1}, (_, i) => a + step * i);
        }
      }
      for (const v of part) {
        const prev = out.at(-1);
        if (out.length && Math.abs(v - prev) <= 1e-9 * Math.max(1, Math.abs(v))) continue;
        if (out.length && v < prev)
          throw new Error(`levels must increase; ${g6(v)} comes after ${g6(prev)}`);
        out.push(v);
      }
    }
    if (out.length < 2) throw new Error("levels: at least two levels make a colour step");
    if (out.length - 1 > MAX_STEPS)
      throw new Error(`levels make ${out.length - 1} steps; at most ${MAX_STEPS} fit`);
    return out;
  }

  // colour.clean: the option set a request names, or null for none
  function cleanColour(text) {
    if (!text) return null;
    const o = JSON.parse(text);
    for (const k of Object.keys(o)) if (o[k] === null || o[k] === "") delete o[k];
    if (!Object.keys(o).length) return null;
    if (o.levels != null) {
      try { o.parsed = parseLevels(o.levels); }
      catch (e) { throw new Error("colour: " + e.message); }
      if (o.bands) throw new Error("colour: levels and bands cannot be combined: " +
        "both set the colour steps; levels=N is bands=N");
      if ((o.norm || "linear") !== "linear") throw new Error(
        `colour: levels and norm=${o.norm} cannot be combined: levels set the steps themselves`);
    }
    return o;
  }

  // colour.centred: the range about `center`, and the ends levels fix
  function centred(lo, hi, o, vmin, vmax) {
    if (o.center != null && !(vmin !== null && vmax !== null)) {
      let half = Math.max(Math.abs(lo - o.center), Math.abs(hi - o.center));
      if (!(half > 0)) half = 1;
      lo = o.center - half; hi = o.center + half;
    }
    if (o.parsed) {
      const [a, b] = levelsEnds(o.parsed);
      if (a !== null) lo = a;
      if (b !== null) hi = b;
    }
    return [lo, hi];
  }

  function extendOf(o) {
    const ext = o.extend || "neither";
    if (!o.parsed) return ext;
    const [l, h] = o.parsed.open || [false, false];
    const low = l || ext === "min" || ext === "both";
    const high = h || ext === "max" || ext === "both";
    return low ? (high ? "both" : "min") : (high ? "max" : "neither");
  }

  // A colour as the page sends it (a hex from its picker, or a CSS name).
  function rgbOf(text) {
    const m = /^#([0-9a-f]{6})$/i.exec(text);
    if (m) return hex(text);
    const g = new OffscreenCanvas(1, 1).getContext("2d");
    g.fillStyle = "#000"; g.fillStyle = text; g.fillRect(0, 0, 1, 1);
    return [...g.getImageData(0, 0, 1, 1).data.slice(0, 3)];
  }
  const hexOf = c => "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");

  // palettes.scale + palettes/encode.indices, painted straight to RGBA
  function colourFrame(q, vals, view, stops, o, lo, hi) {
    const vmin = q.get("vmin") ? +q.get("vmin") : null;
    const vmax = q.get("vmax") ? +q.get("vmax") : null;
    [lo, hi] = centred(lo, hi, o, vmin, vmax);

    // The colormap's own lookup table. The baked ramp samples it at 256
    // points, which for a 256-colour map is the table; a shorter one (GrADS'
    // 13 colours) is read back out of the samples.
    const name = state.meta.cmaps.includes(q.get("cmap")) ? q.get("cmap") : "viridis";
    let N = state.cmapN[name] || 256;
    let lut = N === 256 ? stops.slice()
            : Array.from({length: N}, (_, j) => stops[Math.round((j + 0.5) * 255 / N)]);
    if (o.reverse) lut.reverse();
    const under = o.under_color ? rgbOf(o.under_color) : lut[0];
    const over = o.over_color ? rgbOf(o.over_color) : lut[N - 1];
    const missing = o.missing_color ? rgbOf(o.missing_color) : null;
    const at = t => lut[Math.min(N - 1, Math.max(0, Math.floor(t * N)))];

    let edges = null;
    if (o.parsed) edges = levelEdges(o.parsed, lo, hi);
    else if (o.bands) edges = bandEdges(lo, hi, +o.bands);

    let colourOf, barStops, top = hi;
    if (edges) {
      const nb = edges.length - 1;
      if (nb > N) {                         // more steps than colours: repeat them
        lut = Array.from({length: nb}, (_, i) =>
          lut[Math.min(N - 1, Math.floor(i / (nb - 1) * N))]);
        N = nb;
      }
      // matplotlib's BoundaryNorm: the steps spread over the whole colormap
      const band = Array.from({length: nb}, (_, i) =>
        lut[nb === 1 ? (N - 1) >> 1 : Math.floor((N - 1) / (nb - 1) * i)]);
      top = nextAfter(edges[nb], true);
      const first = edges[0];
      colourOf = v => {
        if (v < first) return under;
        if (v >= top) return over;
        let a = 0, b = nb;                  // the last edge at or below v
        while (b - a > 1) { const m = (a + b) >> 1; if (edges[m] <= v) a = m; else b = m; }
        return band[a];
      };
      barStops = band.map(hexOf);
    } else {
      const kind = o.norm || "linear";
      let f = v => v;
      if (kind === "log") f = v => Math.log(v);
      else if (kind === "symlog") {
        // matplotlib's SymLogNorm, base 10, linscale 1
        const lt = +o.linthresh || 1, adj = 1 / (1 - 1 / 10);
        f = v => Math.abs(v) > lt
          ? Math.sign(v) * lt * (adj + Math.log10(Math.abs(v) / lt)) : v * adj;
      }
      const flo = f(lo), span = f(hi) - flo, gamma = +o.gamma || 1;
      const norm = v => {
        if (kind === "log" && !(v > 0)) return 0;
        const t = (f(v) - flo) / span;
        return kind === "power" ? Math.pow(Math.max(0, t), gamma) : t;
      };
      const steps = N <= MAX_STEPS ? N : MAX_STEPS;
      const pal = Array.from({length: steps}, (_, k) =>
        N <= MAX_STEPS ? lut[k] : at((k + 0.5) / MAX_STEPS));
      colourOf = v => v < lo ? under : v > hi ? over
        : pal[Math.min(steps - 1, Math.max(0, Math.floor(norm(v) * steps)))];
      barStops = Array.from({length: 32}, (_, i) =>
        hexOf(at(Math.min(1, Math.max(0, norm(lo + (hi - lo) * i / 31))))));
    }

    const nx = view.nx, ny = view.ny;
    const img = new ImageData(nx, ny), px = img.data;
    for (let p = 0; p < nx * ny; p++) {
      if (view.blank[p]) continue;
      const v = vals[view.idx[p]];
      const c = Number.isFinite(v) ? colourOf(v) : missing;
      if (!c) continue;
      const k = p * 4;
      px[k] = c[0]; px[k + 1] = c[1]; px[k + 2] = c[2]; px[k + 3] = 255;
    }
    const extend = extendOf(o);
    const spec = {stops: barStops, edges,
                  under: extend === "min" || extend === "both" ? hexOf(under) : null,
                  over: extend === "max" || extend === "both" ? hexOf(over) : null,
                  extend, lo, hi};
    const cv = new OffscreenCanvas(nx, ny);
    cv.getContext("2d").putImageData(img, 0, 0);
    return cv.convertToBlob({type: "image/png"}).then(blob => new Response(blob, {
      status: 200, headers: {"Content-Type": "image/png", "X-Range": `${lo},${hi}`,
                             "X-Colorbar": JSON.stringify(spec)}}));
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
    const view = viewIndex(extent, nx, ny, projOf(q));

    let lo = q.get("vmin") ? +q.get("vmin") : state.fields[name].lo;
    let hi = q.get("vmax") ? +q.get("vmax") : state.fields[name].hi;
    if (!(hi > lo)) hi = lo + 1;

    const stops = await rampOf(q.get("cmap"));
    const opts = cleanColour(q.get("colour"));
    if (opts) return colourFrame(q, vals, view, stops, opts, lo, hi);
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
  // `P` is the page's projection, {proj, plon, plat, psp}, or null on the
  // lon/lat map; bake.py's hook passes it.
  window.GMPAS_OVERLAY = (extent, P) => {
    const el = document.querySelector("#over");
    if (!el || !state.coast) return;
    state.lastOverlay = [extent, P];
    const [x0, x1, y0, y1] = extent;
    const w = el.clientWidth || 1200, h = el.clientHeight || 700;
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const g = cv.getContext("2d");
    const pj = P ? makeProj(P.proj, P.plon, P.plat, P.psp) : null;
    g.strokeStyle = "#111"; g.lineWidth = 0.8; g.beginPath();
    for (const line of (state.world || state.coast)) {
      let pen = false, last = 0;
      for (const [lon, lat] of line) {
        let X = lon, Y = lat;
        if (pj) {
          // the pen lifts where a line leaves the map, or wraps round it
          const xy = pj.fwd(lon, lat), east = wrap180(lon - (+P.plon || 0));
          if (!xy || Math.abs(east - last) > 180) pen = false;
          last = east;
          if (!xy) continue;
          [X, Y] = xy;
        }
        const x = (X - x0) / (x1 - x0) * w, y = (y1 - Y) / (y1 - y0) * h;
        pen ? g.lineTo(x, y) : g.moveTo(x, y);
        pen = true;
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
      if (path === "api/projview") {
        const home = state.proj[q.get("proj") || "auto"];
        return home ? json(home)
                    : json({error: `the demo has no ${q.get("proj")} map`}, 500);
      }
      if (path === "api/unproject") {
        const ll = projOf(q).inv(+q.get("x"), +q.get("y"));
        return json(ll ? {lon: ll[0], lat: ll[1], on: true}
                       : {lon: 0, lat: 0, on: false});
      }
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
