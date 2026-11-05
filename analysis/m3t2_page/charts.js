/* Charts for the Phase 1 reuse study page (analysis/m3t2_report.py).
   Reads window.M3T2 (built from summary.json), draws every figure with
   Observable Plot, and redraws on theme change and resize. Colors come
   from the page's CSS tokens, read at draw time, so both themes work. */
(function () {
  "use strict";
  const D = window.M3T2;
  const root = document.documentElement;
  const tok = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const wl = (w) => tok("--wl-" + w);
  const pol = (p) => tok("--pol-" + p);
  const LABEL = D.labels;
  const POLICY = { ceiling: "No eviction (ceiling)", lru: "LRU", lfu: "LFU",
                   fifo: "FIFO", belady: "Belady (offline best)" };
  const SHAPE = { ceiling: "circle", lru: "circle", lfu: "diamond",
                  fifo: "square", belady: "triangle" };

  const pct = (x, d = 0) => (x == null || isNaN(x)) ? "–" :
    (100 * x).toFixed(d) + "%";
  const secs = (s) => {
    if (s == null || isNaN(s)) return "–";
    if (s < 1) return (1000 * s).toFixed(0) + " ms";
    if (s < 90) return s.toFixed(s < 10 ? 1 : 0) + " s";
    return (s / 60).toFixed(s < 600 ? 1 : 0) + " min";
  };
  const num = (x) => (x == null || isNaN(x)) ? "–" :
    Math.round(x).toLocaleString("en-US");

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  function legend(target, items) {
    const box = el("div", { class: "legend" });
    for (const it of items) {
      const s = el("span");
      if (it.glyph) {
        const g = el("span", { "aria-hidden": "true" }, it.glyph);
        g.style.color = it.color;
        s.append(g, document.createTextNode(it.label));
        box.append(s);
        continue;
      }
      const key = el("i", { class: it.kind === "dot" ? "key-dot" :
        it.kind === "rect" ? "key-rect" : "key-line" });
      key.style.background = it.color;
      if (it.hollow) { key.style.background = "transparent";
        key.style.boxShadow = "inset 0 0 0 2px " + it.color; }
      s.append(key, document.createTextNode(it.label));
      box.append(s);
    }
    target.append(box);
  }

  function tableView(target, columns, rows, caption) {
    const d = el("details", { class: "tableview" });
    d.append(el("summary", {}, caption || "Show the numbers as a table"));
    const wrap = el("div", { class: "tablewrap" });
    const t = el("table");
    const head = el("tr");
    for (const c of columns) head.append(el("th", {}, c));
    const thead = el("thead"); thead.append(head); t.append(thead);
    const body = el("tbody");
    for (const r of rows) {
      const tr = el("tr");
      for (const v of r) tr.append(el("td", {}, v));
      body.append(tr);
    }
    t.append(body); wrap.append(t); d.append(wrap); target.append(d);
  }

  function baseStyle() {
    return { background: "transparent", color: tok("--ink-2"),
             fontFamily: tok("--font-body"), fontSize: "12px",
             overflow: "visible" };
  }

  function logTicks(ticks, fmt) {
    return Plot.text(ticks, { x: (d) => d, text: (d) => fmt(d),
      frameAnchor: "bottom", dy: 15, fill: tok("--ink-2"), fontSize: 12 });
  }

  function width(node) {
    return Math.max(260, Math.min(980, node.clientWidth || 640));
  }

  /* Lines sharing one x grid: a crosshair tooltip lists every series. */
  function lines(node, o) {
    const rows = [];
    for (const s of o.series)
      s.y.forEach((y, i) => { if (y != null && !isNaN(y))
        rows.push({ sid: s.id, label: s.label, x: o.x[i], y }); });
    const tips = o.x.map((x, i) => ({
      x, y: o.yTip ?? (o.yDomain ? o.yDomain[1] : 1),
      title: (o.xFmt || String)(x) + "\n" + o.series.map((s) =>
        s.label + ": " + (o.yFmt || pct)(s.y[i])).join("\n") }));
    const marks = [
      Plot.ruleY([o.yDomain ? o.yDomain[0] : 0], { stroke: tok("--rule") }),
      ...(o.hrules || []).map((r) => Plot.ruleY([r.y], {
        stroke: r.color || tok("--muted"), strokeOpacity: 0.8 })),
      ...(o.hrules || []).map((r) => Plot.text([r], { y: "y", text: "label",
        frameAnchor: "left", dy: -8, dx: 4, fill: tok("--ink-2"),
        fontSize: 11 })),
      ...(o.rules || []).map((r) => Plot.ruleX([r.x], {
        stroke: r.color || tok("--muted"), strokeWidth: 1 })),
      ...(o.rules || []).map((r) => Plot.text([r], { x: "x",
        text: "label", frameAnchor: "top", dy: -6 + (r.dy || 0),
        fill: tok("--ink-2"), fontSize: 11, textAnchor: r.anchor || "start",
        dx: r.anchor === "end" ? -4 : 4 })),
      ...o.series.map((s) => Plot.line(rows.filter((r) => r.sid === s.id), {
        x: "x", y: "y", stroke: s.color, strokeWidth: 2,
        strokeDasharray: s.dash || null, curve: o.curve || "linear" })),
      ...(o.endLabels ? o.series.map((s) => {
        const pts = rows.filter((r) => r.sid === s.id);
        return Plot.text(pts.slice(-1), { x: "x", y: "y",
          text: () => s.short || s.label, dx: 6, dy: s.labelDy || 0,
          textAnchor: "start", fill: tok("--ink"), fontSize: 11 });
      }) : []),
      ...o.series.filter((s) => s.dots).map((s) => Plot.dot(
        rows.filter((r) => r.sid === s.id), { x: "x", y: "y", r: 3.5,
        fill: s.color, stroke: tok("--surface"), strokeWidth: 1.5 })),
      ...(o.xType === "log" && o.xTicks ?
        [logTicks(o.xTicks, o.xTick || String)] : []),
      Plot.ruleX(tips, Plot.pointerX({ x: "x", stroke: tok("--muted") })),
      Plot.tip(tips, Plot.pointerX({ x: "x", y: "y", title: "title",
        anchor: "top-left" })),
    ];
    const logLabels = o.xType === "log" && o.xTicks;
    node.append(Plot.plot({
      width: o.width || width(node), height: o.height || 300,
      marginLeft: 46, marginBottom: logLabels ? 48 : 40, marginTop: 22,
      marginRight: o.marginRight || 18, style: baseStyle(),
      x: { type: o.xType || "linear", domain: o.xDomain, label: o.xLabel,
           tickFormat: logLabels ? () => "" : o.xTick, ticks: o.xTicks,
           labelOffset: logLabels ? 34 : undefined },
      y: { domain: o.yDomain || [0, 1], label: o.yLabel, grid: true,
           tickFormat: o.yTick || ((v) => pct(v)) },
      marks,
    }));
  }

  /* Rows of workloads, one mark per policy (mean), seed range as a bar. */
  function dots(node, o) {
    const rows = [];
    for (const m of o.marks)
      for (const r of o.rows) {
        const v = m.values[r.id];
        if (!v || v.mean == null) continue;
        rows.push({ row: r.label, policy: m.label, id: m.id, x: v.mean,
          lo: v.min, hi: v.max, color: m.color, shape: m.shape,
          hollow: m.hollow, extra: m.extra ? m.extra(r.id) : "" });
      }
    const order = o.rows.map((r) => r.label);
    const gaps = o.gap ? o.rows.map((r) => ({ row: r.label,
      x1: o.gap.from[r.id], x2: o.gap.to[r.id] })).filter((g) =>
      g.x1 != null && g.x2 != null) : [];
    node.append(Plot.plot({
      width: o.width || width(node), height: 40 + 34 * o.rows.length,
      marginLeft: o.marginLeft || 150, marginRight: 24, marginBottom: 36,
      style: baseStyle(),
      x: { domain: o.xDomain || [0, 1], label: o.xLabel, grid: true,
           tickFormat: (v) => (o.fmt || pct)(v) },
      y: { domain: order, label: null },
      marks: [
        Plot.link(gaps, { x1: "x1", x2: "x2", y1: "row", y2: "row",
          stroke: tok("--muted"), strokeOpacity: 0.28, strokeWidth: 7,
          strokeLinecap: "round" }),
        Plot.ruleY(rows, { y: "row", x1: "lo", x2: "hi", stroke: "color",
          strokeOpacity: 0.55, strokeWidth: 1.5 }),
        Plot.dot(rows.filter((r) => r.hollow), { x: "x", y: "row", r: 6,
          stroke: "color", strokeWidth: 2, fill: tok("--surface"),
          symbol: "shape" }),
        Plot.dot(rows.filter((r) => !r.hollow), { x: "x", y: "row", r: 5,
          fill: "color", stroke: tok("--surface"), strokeWidth: 1.5,
          symbol: "shape" }),
        Plot.tip(rows, Plot.pointer({ x: "x", y: "row", title: (d) =>
          d.row + "\n" + d.policy + ": " + (o.fmt ? o.fmt(d.x) : pct(d.x, 1)) +
          (d.lo != null && d.hi != null && d.lo !== d.hi ?
            "  (seeds " + pct(d.lo, 1) + "–" + pct(d.hi, 1) + ")" : "") +
          (d.extra ? "\n" + d.extra : "") })),
      ],
    }));
  }

  /* Horizontal stacked bars (shares or counts). */
  function stacked(node, o) {
    const rows = [];
    for (const r of o.rows)
      for (const s of o.segments) {
        const v = o.value(r.id, s.id);
        if (v == null || isNaN(v)) continue;
        rows.push({ row: r.label, seg: s.label, v, color: s.color,
                    title: r.label + "\n" + s.label + ": " +
                      (o.fmt || pct)(v) });
      }
    node.append(Plot.plot({
      width: o.width || width(node), height: 40 + 30 * o.rows.length,
      marginLeft: o.marginLeft || 150, marginRight: 24, marginBottom: 36,
      style: baseStyle(),
      x: { domain: o.xDomain, label: o.xLabel, grid: true,
           tickFormat: o.xTick || ((v) => pct(v)) },
      y: { domain: o.rows.map((r) => r.label), label: null },
      marks: [
        Plot.barX(rows, Plot.stackX({ x: "v", y: "row", fill: "color",
          insetLeft: 1, insetRight: 1, rx: 2, order: o.segments.map(
            (s) => s.label), z: "seg" })),
        Plot.tip(rows, Plot.pointer(Plot.stackX({ x: "v", y: "row",
          z: "seg", order: o.segments.map((s) => s.label),
          title: "title" }))),
      ],
    }));
  }

  /* A matrix as cells on one sequential hue. */
  function cells(node, o) {
    const rows = [];
    o.rowsLabels.forEach((rl, i) => o.colsLabels.forEach((cl, j) => {
      const v = o.values[i][j];
      rows.push({ r: rl, c: cl, v, title: o.title(i, j) });
    }));
    const ramp = ["--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"]
      .map(tok);
    node.append(Plot.plot({
      width: o.width || Math.min(width(node), 620),
      height: 50 + 34 * o.rowsLabels.length,
      marginLeft: o.marginLeft || 150, marginBottom: 90, marginTop: 10,
      style: baseStyle(),
      x: { domain: o.colsLabels, label: o.xLabel, tickRotate: -35 },
      y: { domain: o.rowsLabels, label: o.yLabel },
      color: { type: "linear", domain: [0, o.max ?? 1], range: [ramp[0],
               ramp[4]], interpolate: "rgb" },
      marks: [
        Plot.cell(rows, { x: "c", y: "r", fill: "v", inset: 1, rx: 3 }),
        Plot.text(rows.filter((d) => d.v >= (o.labelMin ?? 0.05)), {
          x: "c", y: "r", text: (d) => o.fmt(d.v), fontSize: 11,
          fill: (d) => d.v > 0.55 * (o.max ?? 1) ? tok("--seq-1") :
            tok("--ink") }),
        Plot.tip(rows, Plot.pointer({ x: "c", y: "r", title: "title" })),
      ],
    }));
  }

  const W = D.workloads;
  const LADDER = ["--seq-4", "--seq-3", "--seq-2", "--bad"];
  const wlRows = W.map((w) => ({ id: w, label: LABEL[w] }));
  const wlLegend = (node) => legend(node, W.map((w) => ({
    label: LABEL[w], color: wl(w) })));
  const draw = {};

  /* Figures ------------------------------------------------------------ */

  draw["fig-mix"] = (node) => {
    const m = D.mix;
    dots(node, {
      rows: wlRows, xLabel: "Share of the mix (seed average) →",
      xDomain: [0, Math.max(0.55, ...W.map((w) => m[w].tokens))],
      marks: [
        { id: "req", label: "Share of requests", color: tok("--muted"),
          shape: "circle", hollow: true,
          values: Object.fromEntries(W.map((w) => [w, { mean: m[w].requests }])) },
        { id: "tok", label: "Share of prompt tokens", color: tok("--ink"),
          shape: "circle",
          values: Object.fromEntries(W.map((w) => [w, { mean: m[w].tokens }])) },
      ],
    });
    legend(node, [{ label: "Share of requests", color: tok("--muted"),
      kind: "dot", hollow: true }, { label: "Share of prompt tokens",
      color: tok("--ink"), kind: "dot" }]);
  };

  /* ---- The whole mixed workload under each policy ---------------------- */
  const MIXPOLS = ["ceiling", "lru", "lfu", "fifo"];
  const SHORT = { ceiling: "No eviction", lru: "LRU", lfu: "LFU",
                  fifo: "FIFO" };

  draw["fig-mixbars"] = (node) => {
    const M = D.mixRuns;
    if (!M) { node.append(el("p", {}, "Mixed-run data missing.")); return; }
    const rows = MIXPOLS.map((p) => ({ p, label: SHORT[p],
      v: M.bars[p].mean, hi: Math.max(...M.bars[p].seeds) }));
    const seeds = MIXPOLS.flatMap((p) => M.bars[p].seeds.map((v, i) =>
      ({ p, label: SHORT[p], v, seed: M.seeds[i] })));
    node.append(Plot.plot({
      width: width(node), height: 220, marginLeft: 96, marginRight: 64,
      marginBottom: 40, style: baseStyle(),
      x: { domain: [0, 1], grid: true, tickFormat: (v) => pct(v),
           label: "Share of all prompt tokens served from the cache →" },
      y: { domain: rows.map((r) => r.label), label: null },
      marks: [
        ...MIXPOLS.map((p) => Plot.barX(rows.filter((r) => r.p === p), {
          x: "v", y: "label", rx: 3, insetTop: 8, insetBottom: 8,
          fill: p === "ceiling" ? "none" : pol(p),
          stroke: p === "ceiling" ? pol(p) : null,
          strokeDasharray: p === "ceiling" ? "4,3" : null })),
        Plot.dot(seeds, { x: "v", y: "label", r: 4, fill: tok("--surface"),
          stroke: tok("--ink"), strokeWidth: 1.5 }),
        Plot.text(rows, { x: "hi", y: "label", text: (d) => pct(d.v, 1),
          dx: 10, textAnchor: "start", fill: tok("--ink"), fontSize: 13,
          fontWeight: 600 }),
        Plot.tip(seeds, Plot.pointer({ x: "v", y: "label",
          title: (d) => d.label + ", seed " + d.seed + ": " + pct(d.v, 1) })),
      ],
    }));
    legend(node, [
      ...MIXPOLS.slice(1).map((p) => ({ label: SHORT[p] + " (seed mean)",
        color: pol(p), kind: "rect" })),
      { label: "No eviction (ceiling)", color: pol("ceiling"), kind: "rect",
        hollow: true },
      { label: "One seed's run", color: tok("--ink"), kind: "dot",
        hollow: true },
    ]);
  };

  draw["fig-timeline"] = (node) => {
    const M = D.mixRuns; if (!M) return;
    lines(node, {
      x: M.minutes, xDomain: [0, 30], height: 300, marginRight: 84,
      endLabels: true, xTicks: [0, 5, 10, 15, 20, 25, 30],
      xLabel: "Minute of the 30-minute run (when each request was sent) →",
      yLabel: "↑ Prompt tokens served from the cache",
      xFmt: (x) => "minute " + Math.floor(x) + " to " + (Math.floor(x) + 1),
      xTick: (v) => v + " min",
      series: MIXPOLS.map((p) => ({ id: p, label: POLICY[p], short: SHORT[p],
        color: pol(p), y: M.timeline[p], dash: p === "ceiling" ? "5,4" : null,
        labelDy: { ceiling: -2, lru: -11, lfu: 1, fifo: 13 }[p] })),
      rules: [{ x: M.w0, label: "measured from here", color: tok("--muted") }],
    });
    legend(node, MIXPOLS.map((p) => ({ label: POLICY[p], color: pol(p) })));
  };

  draw["fig-mixinterval"] = (node) => {
    const M = D.mixRuns; if (!M) return;
    const box = el("div", { class: "multiples" });
    node.append(box);
    const panel = (title) => {
      const pn = el("div", { class: "panel" });
      pn.append(el("div", { class: "panel-title" }, title));
      const plotNode = el("div", { class: "plot" });
      pn.append(plotNode); box.append(pn); return plotNode;
    };
    const half = Math.max(280, (node.clientWidth || 900) / 2 - 14);
    const a = panel("How long the mix's reuses wait (no-eviction runs)");
    const c = M.intervalCdf;
    const keep = c.x.map((x, i) => [x, c.p[i]]).filter(([x]) => x >= 0.01);
    lines(a, {
      x: keep.map((d) => d[0]), xType: "log", xDomain: [0.01, 1800],
      width: half, height: 260,
      xLabel: "Time since the block's previous use (log) →",
      yLabel: "↑ Share of reuses this soon or sooner",
      xFmt: (x) => "within " + secs(x), xTick: (v) => secs(v),
      xTicks: [0.01, 0.1, 1, 10, 60, 600],
      series: [{ id: "mix", label: "Whole mix", color: tok("--accent"),
                 y: keep.map((d) => d[1]) }],
      rules: [{ x: D.mixRuns.evictP50.lru, label: "LRU keeps ~" +
        secs(D.mixRuns.evictP50.lru), color: pol("lru") }],
    });
    const b = panel("Chance the policy still had the block");
    const I = M.interval;
    lines(b, {
      x: I.mids, xType: "log", xDomain: [0.01, 1800], width: half,
      height: 260, xTicks: [0.01, 0.1, 1, 10, 60, 600],
      xLabel: "Time since the block's previous use (log) →",
      yLabel: "↑ Share of reuses served",
      xFmt: (x) => "about " + secs(x), xTick: (v) => secs(v),
      series: ["lru", "lfu", "fifo"].map((p) => ({ id: p, label: POLICY[p],
        color: pol(p), y: I.served[p], dots: true })),
    });
    legend(node, ["lru", "lfu", "fifo"].map((p) => ({ label: POLICY[p],
      color: pol(p) })).concat([{ label: "Whole mix (left)",
      color: tok("--accent") }]));
  };

  /* Direct labels at each curve's median; policies whose medians are
     within 15% share one label. */
  function ageLabels(M, rows) {
    const groups = [];
    for (const p of ["lru", "fifo", "lfu"]) {
      const g = groups.find((x) => Math.abs(M.evictP50[x[0]] - M.evictP50[p])
        / M.evictP50[p] < 0.15);
      if (g) g.push(p); else groups.push([p]);
    }
    return groups.map((g) => Plot.text(rows.filter((r) => r.p === g[0])
      .filter((r, i, a) => i === Math.floor(a.length * 0.5)), {
      x: "x", y: "y", text: () => g.map((p) => SHORT[p]).join(" and "),
      dx: -10, textAnchor: "end", fill: tok("--ink"), fontSize: 11 }));
  }

  draw["fig-mixages"] = (node) => {
    const M = D.mixRuns; if (!M) return;
    const pols = ["lru", "lfu", "fifo"];
    const rows = pols.flatMap((p) => (M.evictCdf[p].x || []).map((x, i) =>
      ({ p, label: POLICY[p], x: Math.max(x, 0.05), y: M.evictCdf[p].p[i] })));
    node.append(Plot.plot({
      width: width(node), height: 280, marginLeft: 46, marginRight: 60,
      marginBottom: 48, style: baseStyle(),
      x: { type: "log", domain: [0.05, 600], label:
        "How long the block had been idle when it was evicted (log) →",
        tickFormat: () => "", ticks: [0.1, 1, 10, 60, 600],
        labelOffset: 34 },
      y: { domain: [0, 1], grid: true, tickFormat: (v) => pct(v),
           label: "↑ Share of evictions this soon or sooner" },
      marks: [
        Plot.ruleY([0], { stroke: tok("--rule") }),
        ...pols.map((p) => Plot.line(rows.filter((r) => r.p === p), {
          x: "x", y: "y", stroke: pol(p), strokeWidth: 2 })),
        logTicks([0.1, 1, 10, 60, 600], secs),
        ...ageLabels(M, rows),
        Plot.tip(rows, Plot.pointer({ x: "x", y: "y", title: (d) =>
          d.label + ": " + pct(d.y, 0) + " of evictions within " +
          secs(d.x) })),
      ],
    }));
    legend(node, pols.map((p) => ({ label: POLICY[p] + ", median " +
      secs(M.evictP50[p]), color: pol(p) })));
  };

  draw["fig-mixlost"] = (node) => {
    const segs = [
      { id: "kept", label: "Reused", color: tok("--seq-4") },
      { id: "evicted", label: "Lost: evicted", color: tok("--seq-2") },
      { id: "cut_off", label: "Lost: cached but cut off", color: tok("--bad") },
    ];
    stacked(node, {
      rows: ["lru", "lfu", "fifo"].map((p) => ({ id: p, label: SHORT[p] })),
      xDomain: [0, 1], marginLeft: 70, segments: segs,
      xLabel: "Share of the blocks the no-eviction run reuses →",
      value: (p, sid) => { const L = D.lost[p].all;
        const t = L.kept + L.evicted + L.cut_off; return t ? L[sid] / t : null; },
    });
    legend(node, segs.map((g) => ({ label: g.label, color: g.color,
                                     kind: "rect" })));
  };

  draw["fig-headline"] = (node) => {
    const h = D.headline;
    const pols = ["ceiling", "lru", "lfu", "fifo"];
    dots(node, {
      rows: [{ id: "all", label: "Whole mix" }, ...wlRows],
      xLabel: "Prompt tokens served from the cache (usable reuse) →",
      gap: { from: h.lru, to: h.ceiling },
      marks: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p),
        shape: SHAPE[p], hollow: p === "ceiling",
        values: Object.fromEntries(Object.entries(h[p]).map(([w, v]) =>
          [w, v.usable])) })),
    });
    legend(node, pols.map((p) => ({ label: POLICY[p], color: pol(p),
      kind: "dot", hollow: p === "ceiling" })));
  };

  draw["fig-comeback"] = (node) => {
    const c = D.comeBack;
    const rules = Object.entries(D.retention).filter(([p]) => p !== "lfu")
      .map(([p, q], i) => ({ x: q.p50, label: POLICY[p] + " evicts ~" +
        secs(q.p50), color: pol(p), dy: 14 * i }));
    lines(node, {
      x: c.grid, xType: "log", xDomain: [0.05, 1200],
      xLabel: "Time since the block was freed (log scale) →",
      yLabel: "↑ Share looked up again",
      xFmt: (x) => "after " + secs(x), xTick: (v) => secs(v),
      series: W.map((w) => ({ id: w, label: LABEL[w], color: wl(w),
                              y: c.series[w] })),
      rules,
    });
    wlLegend(node);
  };

  draw["fig-hitinterval"] = (node) => {
    const h = D.hitByInterval;
    const mids = h.mids;
    const box = el("div", { class: "multiples" });
    for (const p of ["lru", "lfu", "fifo"]) {
      const panel = el("div", { class: "panel" });
      panel.append(el("div", { class: "panel-title" }, POLICY[p]));
      const plotNode = el("div", { class: "plot" });
      panel.append(plotNode);
      box.append(panel);
      node.append(box);
      lines(plotNode, {
        x: mids, xType: "log", xDomain: [0.05, 2000], height: 240,
        width: Math.max(260, (node.clientWidth || 900) / 3 - 14),
        xLabel: "Reuse interval (log) →", yLabel: "↑ Usable hit",
        xFmt: (x) => "interval ~" + secs(x), xTick: (v) => secs(v),
        xTicks: 4,
        series: W.map((w) => ({ id: w, label: LABEL[w], color: wl(w),
                                y: h[p][w], dots: true })),
      });
    }
    wlLegend(node);
  };

  draw["fig-distance"] = (node) => {
    const d = D.distance;
    stacked(node, {
      rows: wlRows, xDomain: [0, 1],
      xLabel: "Share of reuses (no-eviction run) →",
      segments: [
        { id: "within", label: "Within the cache's 8,192 blocks",
          color: tok("--seq-3") },
        { id: "beyond", label: "Beyond them: LRU cannot keep these",
          color: tok("--seq-1") },
      ],
      value: (w, s) => s === "within" ? 1 - d[w] : d[w],
    });
    legend(node, [{ label: "Reuse distance below the cache's 8,192 blocks",
      color: tok("--seq-3"), kind: "rect" }, { label: "Beyond the cache",
      color: tok("--seq-1"), kind: "rect" }]);
  };

  draw["fig-sources"] = (node) => {
    const s = D.sources;
    const segs = [
      { id: "same_session", label: "Same conversation", color: tok("--seq-4") },
      { id: "other_session", label: "Another conversation, same workload",
        color: tok("--seq-2") },
      { id: "other_workload", label: "Another workload", color: tok("--seq-1") },
    ];
    stacked(node, { rows: wlRows, xDomain: [0, 1], segments: segs,
      xLabel: "Share of reused blocks, by who used them before →",
      value: (w, sid) => s[w][sid] });
    legend(node, segs.map((g) => ({ label: g.label, color: g.color,
                                     kind: "rect" })));
  };

  draw["fig-lost"] = (node) => {
    const box = el("div", { class: "multiples" });
    node.append(box);
    const segs = [
      { id: "kept", label: "Reused", color: tok("--seq-4") },
      { id: "evicted", label: "Lost: evicted", color: tok("--seq-2") },
      { id: "cut_off", label: "Lost: cached but cut off", color: tok("--bad") },
    ];
    for (const p of ["lru", "lfu", "fifo"]) {
      const panel = el("div", { class: "panel" });
      panel.append(el("div", { class: "panel-title" }, POLICY[p]));
      const plotNode = el("div", { class: "plot" });
      panel.append(plotNode); box.append(panel);
      const L = D.lost[p];
      stacked(plotNode, {
        rows: wlRows, xDomain: [0, 1], marginLeft: 120,
        width: Math.max(260, (node.clientWidth || 900) / 3 - 14),
        xLabel: "Share of blocks the ceiling reuses →", segments: segs,
        value: (w, sid) => { const t = L[w].kept + L[w].evicted + L[w].cut_off;
                             return t ? L[w][sid] / t : null; },
      });
    }
    legend(node, segs.map((g) => ({ label: g.label, color: g.color,
                                     kind: "rect" })));
  };

  draw["fig-position"] = (node) => {
    const p = D.position;
    lines(node, {
      x: p.mids, xType: "log", xDomain: [0.7, 2048], height: 280,
      xLabel: "Block position in the prompt (log) →",
      yLabel: "↑ Reused with no eviction",
      xFmt: (x) => "blocks ~" + Math.round(x), xTick: (v) => String(v),
      series: W.map((w) => ({ id: w, label: LABEL[w], color: wl(w),
                              y: p.ceiling[w] })),
    });
    wlLegend(node);
  };

  draw["fig-density"] = (node) => {
    const d = D.density.lru;
    dots(node, {
      rows: wlRows, xLabel: "Share under LRU →",
      xDomain: [0, Math.max(0.9, ...W.map((w) => Math.max(d[w].pool,
                                                         d[w].revive)))],
      gap: { from: Object.fromEntries(W.map((w) => [w, d[w].pool])),
             to: Object.fromEntries(W.map((w) => [w, d[w].revive])) },
      marks: [
        { id: "pool", label: "Share of idle cache time", color: tok("--muted"),
          shape: "square", values: Object.fromEntries(W.map((w) =>
            [w, { mean: d[w].pool }])) },
        { id: "revive", label: "Share of reuses from idle blocks",
          color: tok("--accent"), shape: "circle", values: Object.fromEntries(
            W.map((w) => [w, { mean: d[w].revive }])) },
      ],
    });
    legend(node, [{ label: "Share of the time idle blocks spend in the cache",
      color: tok("--muted"), kind: "rect" }, { label: "Share of reuses " +
      "served from idle blocks", color: tok("--accent"), kind: "dot" }]);
  };

  draw["fig-evictions"] = (node) => {
    const box = el("div", { class: "multiples" });
    node.append(box);
    for (const p of ["lru", "lfu", "fifo"]) {
      const e = D.evictions[p];
      const panel = el("div", { class: "panel" });
      panel.append(el("div", { class: "panel-title" }, POLICY[p]));
      const plotNode = el("div", { class: "plot" });
      panel.append(plotNode); box.append(panel);
      const total = e.counts.flat().reduce((a, b) => a + b, 0) || 1;
      const vals = e.counts.map((r) => r.map((v) => v / total));
      const lab = W.map((w) => LABEL[w]);
      cells(plotNode, {
        rowsLabels: lab, colsLabels: lab, values: vals, marginLeft: 120,
        width: Math.max(280, (node.clientWidth || 900) / 3 - 14),
        max: Math.max(...vals.flat()), labelMin: 0.04,
        xLabel: "Workload evicted →", yLabel: null,
        fmt: (v) => pct(v), title: (i, j) => lab[i] + " evicted " + lab[j] +
          ": " + pct(vals[i][j], 1) + " of all evictions",
      });
    }
  };

  draw["fig-ladder"] = (node) => {
    const L = D.identify && D.identify.ladder;
    if (!L) { node.append(el("p", {}, "Identification report missing."));
              return; }
    stacked(node, {
      rows: L.rows.map((r) => ({ id: r.id, label: LABEL[r.id] || r.label })),
      xDomain: [0, 1], xLabel: "Share of test requests →",
      segments: L.segments.map((s, i) => ({ id: s.id, label: s.label,
        color: tok(LADDER[i]) })),
      value: (rid, sid) => L.values[rid][sid],
    });
    legend(node, L.segments.map((s, i) => ({ label: s.label, kind: "rect",
      color: tok(LADDER[i]) })));
  };

  draw["fig-confusion"] = (node) => {
    const C = D.identify && D.identify.confusion;
    if (!C) return;
    const lab = C.labels.map((w) => LABEL[w] || w);
    cells(node, {
      rowsLabels: lab, colsLabels: C.predicted.map((w) => LABEL[w] || w),
      values: C.share, max: 1, labelMin: 0.03, fmt: (v) => pct(v),
      xLabel: "Predicted workload →", yLabel: null,
      title: (i, j) => lab[i] + " predicted as " + (LABEL[C.predicted[j]] ||
        C.predicted[j]) + ": " + pct(C.share[i][j], 1),
    });
  };

  draw["fig-signals"] = (node) => {
    const S = D.signals;
    if (!S) { node.append(el("p", {}, "Signals report missing.")); return; }
    const ages = S.landmarks;
    const ageLabel = (a) => a === 0 ? "just freed" :
      "idle " + (a < 60 ? a + " s" : a / 60 + " min");
    // High-contrast ramp steps first; the last age is drawn in grey.
    const ramp = ["--seq-5", "--seq-4", "--seq-3", "--muted"];
    const glyphs = ["●", "◆", "■", "▲"];
    const rows = S.rows.filter((r) => r.id !== "idle_age");
    dots(node, {
      rows: rows.map((r) => ({ id: r.id, label: r.label.replace(/^\+ /, "") })),
      xDomain: [0.4, 1], marginLeft: 220, fmt: (v) => v.toFixed(2),
      xLabel: "Ranks blocks that come back within a minute (AUC) →",
      marks: ages.map((a, i) => ({ id: "a" + a, label: ageLabel(a),
        color: tok(ramp[i]), shape: ["circle", "diamond", "square",
          "triangle"][i], values: Object.fromEntries(rows.map((r) => [r.id,
          { mean: S.by_age[r.id][String(a)] }])) })),
    });
    legend(node, ages.map((a, i) => ({ label: "Blocks " + ageLabel(a),
      glyph: glyphs[i], color: tok(ramp[i]) })));
  };

  draw["fig-capacity"] = (node) => {
    const c = D.capacity;
    if (!c) { node.append(el("p", {}, "Simulator sweep missing.")); return; }
    const pols = ["belady", "lru", "lfu", "fifo"];
    lines(node, {
      x: c.capacities, xType: "log", xDomain: [c.capacities[0],
        c.capacities[c.capacities.length - 1]], height: 300,
      xLabel: "Cache size in tokens (log; 1K = 1,024) →",
      yLabel: "↑ Usable reuse (cache replay)",
      xFmt: (x) => num(x) + " tokens",
      xTick: (v) => Math.round(v / 1024) + "K", xTicks: c.capacities,
      yDomain: [0, Math.max(0.8, c.ceiling + 0.02)],
      rules: [{ x: c.fixed, label: "the fixed cache: 131,072 tokens",
                color: tok("--muted") }],
      hrules: [{ y: c.ceiling, label: "no eviction " + pct(c.ceiling) }],
      series: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p),
                                 y: c.curves[p], dots: true })),
    });
    legend(node, pols.map((p) => ({ label: POLICY[p], color: pol(p) }))
      .concat([{ label: "No-eviction ceiling " + pct(c.ceiling),
                 color: tok("--muted") }]));
  };

  draw["fig-belady"] = (node) => {
    const b = D.beladySplit;
    if (!b) return;
    dots(node, {
      rows: wlRows,
      xLabel: "Usable reuse with a 131,072-token cache (cache replay) →",
      gap: { from: b.lru, to: b.belady },
      marks: ["ceiling", "belady", "lru"].map((p) => ({ id: p,
        label: POLICY[p], color: pol(p), shape: SHAPE[p],
        hollow: p === "ceiling", values: Object.fromEntries(W.map((w) =>
          [w, { mean: b[p][w] }])) })),
    });
    legend(node, ["ceiling", "belady", "lru"].map((p) => ({
      label: POLICY[p], color: pol(p), kind: "dot",
      hollow: p === "ceiling" })));
  };

  /* Tables under each figure (the same numbers, readable without hover). */
  const tables = {
    "fig-mixbars": () => D.mixRuns ? [["Policy", "Seed mean",
      ...D.mixRuns.seeds.map((k) => "Seed " + k), "Share of the ceiling"],
      MIXPOLS.map((p) => [POLICY[p], pct(D.mixRuns.bars[p].mean, 1),
        ...D.mixRuns.bars[p].seeds.map((v) => pct(v, 1)),
        p === "ceiling" ? "100%" : pct(D.mixRuns.efficiency[p], 1)])] : null,
    "fig-timeline": () => D.mixRuns ? [["Minute", ...MIXPOLS.map((p) =>
      SHORT[p])], D.mixRuns.minutes.map((m, i) => [Math.floor(m) + "–" +
      (Math.floor(m) + 1), ...MIXPOLS.map((p) =>
        pct(D.mixRuns.timeline[p][i], 1))])] : null,
    "fig-mixinterval": () => D.mixRuns ? [["Time since previous use",
      "LRU served", "LFU served", "FIFO served"],
      D.mixRuns.interval.edges.slice(0, -1).map((e, i) => [secs(e) + "–" +
        secs(D.mixRuns.interval.edges[i + 1]), ...["lru", "lfu", "fifo"]
        .map((p) => pct(D.mixRuns.interval.served[p][i], 0))])] : null,
    "fig-mixages": () => D.mixRuns ? [["Policy", "10% of evictions within",
      "50% within", "90% within"], ["lru", "lfu", "fifo"].map((p) => [
      POLICY[p], secs(D.retention[p].p10), secs(D.retention[p].p50),
      secs(D.retention[p].p90)])] : null,
    "fig-mixlost": () => [["Policy", "Reused (blocks per seed)", "Evicted",
      "Cut off"], ["lru", "lfu", "fifo"].map((p) => { const L = D.lost[p].all;
        return [POLICY[p], num(L.kept), num(L.evicted), num(L.cut_off)]; })],
    "fig-mix": () => [["Workload", "Loop", "Requests", "Prompt tokens",
      "Sessions", "Pause p50", "Kept of trace"], W.map((w) => [LABEL[w],
      D.loop[w], pct(D.mix[w].requests, 1), pct(D.mix[w].tokens, 1),
      num(D.mix[w].sessions), D.mix[w].pause50 == null ? "recorded" :
        secs(D.mix[w].pause50), D.mix[w].kept == null ? "–" :
        pct(D.mix[w].kept, 1)])],
    "fig-headline": () => [["Workload", "Ceiling", "LRU", "LFU", "FIFO",
      "LFU − LRU", "FIFO − LRU"], ["all", ...W].map((w) => [w === "all" ?
      "Whole mix" : LABEL[w], ...["ceiling", "lru", "lfu", "fifo"].map((p) =>
      pct(D.headline[p][w].usable.mean, 1)), ...["lfu", "fifo"].map((p) => {
        const v = D.headline[p][w].vs_lru; return v ? (v.mean >= 0 ? "+" : "") +
          (100 * v.mean).toFixed(1) + " pts" : "–"; })])],
    "fig-comeback": () => [["Workload", "15 s", "1 min", "5 min", "10 min",
      "Interval p50", "Interval p90"], W.map((w) => {
        const g = D.comeBack.grid, s = D.comeBack.series[w];
        const at = (t) => pct(s[g.indexOf(t)], 0);
        return [LABEL[w], at(15), at(60), at(300), at(600),
          secs(D.intervals[w].p50), secs(D.intervals[w].p90)]; })],
    "fig-hitinterval": () => [["Workload", "Policy", ...D.hitByInterval.edges
      .slice(0, -1).map((e, i) => secs(e) + "–" + secs(D.hitByInterval.edges[i + 1]))],
      ["lru", "lfu", "fifo"].flatMap((p) => W.map((w) => [LABEL[w], POLICY[p],
        ...D.hitByInterval[p][w].map((v) => pct(v, 0))]))],
    "fig-distance": () => [["Workload", "Reuses beyond 8,192 blocks"],
      W.map((w) => [LABEL[w], pct(D.distance[w], 1)])],
    "fig-sources": () => [["Workload", "Same conversation", "Other conversation",
      "Other workload", "Read by a first turn"], W.map((w) => [LABEL[w],
      pct(D.sources[w].same_session, 1), pct(D.sources[w].other_session, 1),
      pct(D.sources[w].other_workload, 1), pct(D.sources[w].first_turn, 1)])],
    "fig-lost": () => [["Workload", "Policy", "Reused (blocks per seed)",
      "Evicted", "Cut off"],
      ["lru", "lfu", "fifo"].flatMap((p) => W.map((w) => {
        const L = D.lost[p][w]; return [LABEL[w], POLICY[p], num(L.kept),
          num(L.evicted), num(L.cut_off)]; }))],
    "fig-position": () => [["Workload", ...D.position.labels], W.map((w) =>
      [LABEL[w], ...D.position.ceiling[w].map((v) => pct(v, 0))])],
    "fig-density": () => [["Workload", "Idle cache time", "Reuses from idle",
      "Reuses per block-minute"], W.map((w) => [LABEL[w],
      pct(D.density.lru[w].pool, 1), pct(D.density.lru[w].revive, 1),
      D.density.lru[w].rate == null ? "–" : D.density.lru[w].rate.toFixed(2)])],
    "fig-evictions": () => [["Evictions per seed: evicting → evicted",
      ...W.map((w) => LABEL[w])],
      ["lru", "lfu", "fifo"].flatMap((p) => W.map((w, i) =>
        [POLICY[p] + ": " + LABEL[w], ...D.evictions[p].counts[i].map(num)]))],
    "fig-ladder": () => D.identify ? [["Workload", "First block",
      "First two blocks", "Decision tree", "Wrong"], D.identify.ladder.rows
      .map((r) => [LABEL[r.id] || r.label, ...["b1", "f2", "tree", "wrong"]
        .map((k) => pct(D.identify.ladder.values[r.id][k], 1))])] : null,
    "fig-confusion": () => D.identify ? [["True \\ predicted", ...W.map((w) =>
      LABEL[w])], W.map((w, i) => [LABEL[w], ...D.identify.confusion.share[i]
        .map((v) => pct(v, 1))])] : null,
    "fig-signals": () => D.signals ? [["Signals (AUC, back within a minute)",
      ...D.signals.landmarks.map((a) => a === 0 ? "just freed" :
        "idle " + (a < 60 ? a + " s" : a / 60 + " min"))],
      D.signals.rows.map((r) => [r.label,
      ...D.signals.landmarks.map((a) =>
        D.signals.by_age[r.id][String(a)].toFixed(3))])] : null,
    "fig-capacity": () => D.capacity ? [["Cache (tokens)", "Belady", "LRU",
      "LFU", "FIFO"], D.capacity.capacities.map((c, i) => [num(c),
      ...["belady", "lru", "lfu", "fifo"].map((p) =>
        pct(D.capacity.curves[p][i], 1))])] : null,
    "fig-belady": () => D.beladySplit ? [["Workload", "Ceiling", "Belady",
      "LRU"], W.map((w) => [LABEL[w], ...["ceiling", "belady", "lru"].map(
        (p) => pct(D.beladySplit[p][w], 1))])] : null,
  };

  function renderAll() {
    for (const [id, fn] of Object.entries(draw)) {
      const node = document.getElementById(id);
      if (!node) continue;
      node.replaceChildren();
      const plotNode = el("div", { class: "plot" });
      node.append(plotNode);
      try { fn(plotNode); } catch (err) {
        plotNode.append(el("p", {}, "This chart failed to draw: " + err));
      }
      const t = tables[id] && tables[id]();
      if (t) tableView(node, t[0], t[1]);
    }
  }

  let pending = null;
  const later = () => { clearTimeout(pending); pending = setTimeout(renderAll, 120); };
  renderAll();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", later);
  new MutationObserver(later).observe(root, { attributes: true,
    attributeFilter: ["data-theme"] });
  let lastWidth = document.body.clientWidth;
  new ResizeObserver(() => { const w = document.body.clientWidth;
    if (Math.abs(w - lastWidth) > 40) { lastWidth = w; later(); } })
    .observe(document.body);
})();
