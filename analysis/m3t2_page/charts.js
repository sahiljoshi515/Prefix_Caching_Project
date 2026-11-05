/* Charts for the mixed-workload eviction study (analysis/m3t2_report.py).
   Reads window.M3T2 (built from summary.json), draws each figure with
   Observable Plot, and redraws on theme change and resize. Colours come
   from the page's CSS tokens at draw time, so both themes work. Policies
   carry the colours; workloads are named in text, never by colour. */
(function () {
  "use strict";
  const D = window.M3T2;
  const root = document.documentElement;
  const tok = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const pol = (p) => tok("--pol-" + p);
  const LABEL = D.labels;
  const POLICY = D.policyLabels;
  const SHAPE = { lru: "circle", lfu: "diamond", fifo: "square",
                  ttdf: "cross", race_rp: "star", sampled_lrfu: "wye",
                  belady: "triangle" };

  const pct = (x, d = 0) => (x == null || isNaN(x)) ? "–" :
    (100 * x).toFixed(d) + "%";
  const pts = (x) => (x == null || isNaN(x)) ? "–" :
    (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(1) + " pts";
  const num = (x) => (x == null || isNaN(x)) ? "–" :
    Math.round(x).toLocaleString("en-US");
  const secs = (s) => {
    if (s == null || isNaN(s)) return "–";
    if (s < 1) return (1000 * s).toFixed(0) + " ms";
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + " s";
    return (s / 60).toFixed(s < 600 ? 1 : 0) + " min";
  };
  const tokens = (n) => n >= 1048576 ? (n / 1048576) + "M" : (n / 1024) + "K";
  const tickSecs = (s) => s < 60 ? s + " s" : (s / 60) + " min";
  // Below this chart width, axes thin or tilt their tick labels.
  const NARROW = 560;

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  /* A legend placed above the chart that `node` will hold. */
  function legend(node, items) {
    const box = el("div", { class: "legend" });
    for (const it of items) {
      const s = el("span");
      if (it.glyph) {
        s.append(el("b", { class: "key-glyph", "aria-hidden": "true" },
          it.glyph), document.createTextNode(it.label));
        box.append(s);
        continue;
      }
      const key = el("i", { class: "key-" + (it.kind || "rect") });
      if (it.kind === "dash" || it.kind === "line")
        key.style.borderTopColor = it.color;
      else if (it.hollow) key.style.boxShadow = "inset 0 0 0 2px " + it.color;
      else key.style.background = it.color;
      s.append(key, document.createTextNode(it.label));
      box.append(s);
    }
    node.before(box);
  }

  const CEILING_KEY = () => ({ label: "No eviction (ceiling)",
    color: tok("--ink"), kind: "dash" });
  const policyLegend = (node, pols, extra) => legend(node, [
    ...pols.map((p) => ({ label: POLICY[p], color: pol(p) })),
    ...(extra || [])]);

  function tableView(target, columns, rows) {
    const d = el("details", { class: "tableview" });
    d.append(el("summary", {}, "Show the numbers as a table"));
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

  function width(node) {
    return Math.max(280, Math.min(980, node.clientWidth || 640));
  }

  /* d3's log formatter blanks labels that are not powers of ten, so log
     axes draw their own labels as text marks. */
  function logTicks(ticks, fmt) {
    return Plot.text(ticks, { x: (d) => d, text: (d) => fmt(d),
      frameAnchor: "bottom", dy: 15, fill: tok("--ink-2"), fontSize: 12 });
  }

  /* Lines on a shared x grid; a crosshair tooltip lists every series. */
  function lines(node, o) {
    const rows = [];
    for (const s of o.series)
      s.y.forEach((y, i) => { if (y != null && !isNaN(y))
        rows.push({ sid: s.id, x: o.x[i], y }); });
    const tips = o.x.map((x, i) => ({ x, y: o.yDomain[1],
      title: (o.xFmt || String)(x) + "\n" + o.series.map((s) =>
        s.label + ": " + pct(s.y[i], 1)).join("\n") }));
    const log = o.xType === "log";
    const marks = [
      Plot.ruleY([o.yDomain[0]], { stroke: tok("--rule") }),
      ...(o.hrules || []).flatMap((r) => [
        Plot.ruleY([r.y], { stroke: tok("--ink"), strokeDasharray: "4,3",
          strokeWidth: 1.5 }),
        Plot.text([r], { y: "y", text: "label", frameAnchor: "left",
          dx: 4, dy: -8, fill: tok("--ink"), fontSize: 11 }),
      ]),
      ...(o.vrules || []).flatMap((r) => [
        Plot.ruleX([r.x], { stroke: tok("--muted"), strokeWidth: 1 }),
        Plot.text([r], { x: "x", text: "label", frameAnchor: "bottom",
          dy: -6, dx: 4, textAnchor: "start", fill: tok("--ink-2"),
          fontSize: 11 }),
      ]),
      ...o.series.map((s) => Plot.line(rows.filter((r) => r.sid === s.id), {
        x: "x", y: "y", stroke: s.color, strokeWidth: 2,
        strokeDasharray: s.dash || null })),
      ...o.series.map((s) => Plot.dot(rows.filter((r) => r.sid === s.id), {
        x: "x", y: "y", r: 3.5, fill: s.color, stroke: tok("--surface"),
        strokeWidth: 1.5, symbol: SHAPE[s.id] || "circle" })),
      ...(o.endLabels ? o.series.map((s) => Plot.text(
        rows.filter((r) => r.sid === s.id).slice(-1), { x: "x", y: "y",
          text: () => s.label, dx: 8, dy: s.labelDy || 0,
          textAnchor: "start", fill: tok("--ink"), fontSize: 11 })) : []),
      ...(log ? [logTicks(o.xTicks, o.xTick)] : []),
      ...(o.extra || []),
      Plot.ruleX(tips, Plot.pointerX({ x: "x", stroke: tok("--muted") })),
      Plot.tip(tips, Plot.pointerX({ x: "x", y: "y", title: "title",
        anchor: "top-left" })),
    ];
    node.append(Plot.plot({
      width: width(node), height: o.height || 300,
      marginLeft: 46, marginBottom: log ? 48 : 40, marginTop: 22,
      marginRight: o.marginRight || 20, style: baseStyle(),
      x: { type: o.xType || "linear", domain: o.xDomain, label: o.xLabel,
           ticks: o.xTicks, tickFormat: log ? () => "" : o.xTick,
           labelOffset: log ? 36 : undefined },
      y: { domain: o.yDomain, label: o.yLabel, grid: true,
           tickFormat: (v) => pct(v) },
      marks,
    }));
  }

  /* Horizontal bars grouped by row, one bar per series: the seed mean, the
     seed range as a thin line, the value at the bar's end, and an optional
     reference (the no-eviction ceiling) as a dashed tick across the row. */
  function groupedBars(node, o) {
    const data = [];
    for (const r of o.rows)
      for (const s of o.series) {
        const v = o.value(r.id, s.id);
        if (!v || v.mean == null) continue;
        data.push({ row: r.label, s: s.label, v: v.mean,
          lo: v.min ?? v.mean, hi: v.max ?? v.mean, color: s.color });
      }
    const refs = o.ref ? o.rows.map((r) => ({ row: r.label,
      v: (o.ref(r.id) || {}).mean })).filter((d) => d.v != null) : [];
    const fmt = o.fmt || ((v) => pct(v, o.digits ?? 1));
    node.append(Plot.plot({
      width: width(node), height: o.rows.length * (o.series.length * 20 + 16)
        + 44, marginLeft: o.marginLeft || 108, marginRight: 58,
      marginTop: 6, marginBottom: 36, style: baseStyle(),
      fy: { domain: o.rows.map((r) => r.label), label: null, axis: "left",
            tickSize: 0, padding: 0.16 },
      y: { domain: o.series.map((s) => s.label), axis: null, padding: 0.14 },
      x: { domain: o.xDomain || [0, 1], label: o.xLabel, grid: true,
           tickFormat: o.tickFormat || ((v) => pct(v)) },
      marks: [
        Plot.barX(data, { fy: "row", y: "s", x: "v", fill: "color", rx: 2 }),
        Plot.ruleY(data.filter((d) => d.hi > d.lo), { fy: "row", y: "s",
          x1: "lo", x2: "hi", stroke: tok("--ink"), strokeOpacity: 0.6,
          strokeWidth: 1.5 }),
        Plot.text(data, { fy: "row", y: "s", x: (d) => Math.max(d.v, d.hi),
          text: (d) => fmt(d.v), dx: 5, textAnchor: "start",
          fill: tok("--ink"), fontSize: 11 }),
        Plot.tickX(refs, { fy: "row", x: "v", stroke: tok("--ink"),
          strokeWidth: 2, strokeDasharray: "4,3" }),
        // A reference per series (one bar of each row): a dashed tick.
        Plot.tickX(o.seriesRefs ? o.rows.flatMap((r) => o.seriesRefs.map(
          (x) => ({ row: r.label, s: x.s, v: x.v }))) : [], { fy: "row",
          y: "s", x: "v", stroke: tok("--ink"), strokeWidth: 2,
          strokeDasharray: "3,2" }),
        Plot.tip(data, Plot.pointer({ fy: "row", y: "s", x: "v",
          title: (d) => d.row + " · " + d.s + ": " + fmt(d.v) +
            (d.hi > d.lo ? "\nseeds " + fmt(d.lo) + " to " +
              fmt(d.hi) : "") })),
      ],
    }));
  }

  /* A matrix as cells on one sequential hue, labelled in each cell. */
  function cells(node, o) {
    const rows = [];
    o.rowLabels.forEach((rl, i) => o.colLabels.forEach((cl, j) => {
      rows.push({ r: rl, c: cl, v: o.values[i][j], title: o.title(i, j) });
    }));
    const tilt = width(node) < NARROW;  // long names would run together
    node.append(Plot.plot({
      width: Math.min(width(node), 560), height: 70 + 48 * o.rowLabels.length +
        (tilt ? 24 : 0),
      marginLeft: 104, marginBottom: tilt ? 70 : 46, marginTop: 8,
      style: baseStyle(),
      x: { domain: o.colLabels, label: o.xLabel, axis: "bottom",
           tickSize: 0, tickRotate: tilt ? -35 : 0 },
      y: { domain: o.rowLabels, label: o.yLabel, tickSize: 0 },
      color: { type: "linear", domain: [0, 1],
               range: [tok("--seq-1"), tok("--seq-5")], interpolate: "rgb" },
      marks: [
        Plot.cell(rows, { x: "c", y: "r", fill: "v", inset: 1.5, rx: 3 }),
        Plot.text(rows, { x: "c", y: "r", text: (d) => pct(d.v, 0),
          fontSize: 12, fill: (d) => d.v > 0.5 ? tok("--seq-1") :
            tok("--ink") }),
        Plot.tip(rows, Plot.pointer({ x: "c", y: "r", title: "title" })),
      ],
    }));
  }

  /* Reuse-interval bins as points: the geometric middle of each bin with
     enough reuses on offer under every policy. */
  const INTERVAL_TICKS = [0.1, 1, 10, 60, 300, 1200];
  const binLabel = (b) => b.hi == null ? "over " + secs(b.lo) :
    (b.lo <= 0 ? "under " + secs(b.hi) : secs(b.lo) + " to " + secs(b.hi));

  function intervalChart(node, curves, pols, height) {
    const bins = curves[pols[0]];
    const keep = bins.map((b, i) => i).filter((i) => bins[i].hi != null &&
      pols.every((p) => curves[p][i].offered >= D.minOffered));
    const mid = (b) => b.lo <= 0 ? b.hi / 2 : Math.sqrt(b.lo * b.hi);
    lines(node, {
      x: keep.map((i) => mid(bins[i])), xType: "log",
      xTicks: width(node) < NARROW ? INTERVAL_TICKS.filter((t) => t !== 300)
        : INTERVAL_TICKS, xTick: tickSecs, xDomain: [0.03, 1800],
      xFmt: (v) => "Waited " + binLabel(bins[keep.find((i) =>
        mid(bins[i]) === v)]),
      xLabel: "Time the block sat unused before it was needed again →",
      yDomain: [0, 1], yLabel: "↑ Share of those reuses served from cache",
      height: height || 300,
      series: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p),
        y: keep.map((i) => curves[p][i].share) })),
    });
  }
  const intervalTable = (curves, prefix, pols = P) => [[...(prefix ?
    ["Workload"] : []), "Block unused for", "Reuses on offer",
    ...pols.map((p) => POLICY[p])],
    curves[pols[0]].map((b, i) => [...(prefix ? [prefix] : []), binLabel(b),
      num(b.offered), ...pols.map((p) => pct(curves[p][i].share, 1))])
      .filter((r) => r[prefix ? 2 : 1] !== "0")];

  const draw = {};
  const tables = {};
  const P = D.policies;
  // The mixed-workload engine views add TTDF and RACE-RP once they ran.
  const M = D.mixedPolicies || P;

  /* 1. The mix: share of reuses served, by how long the block waited. */
  draw["fig-q1"] = (node) => {
    policyLegend(node, M);
    intervalChart(node, D.q1.curves, M, 320);
  };
  tables["fig-q1"] = () => intervalTable(D.q1.curves, null, M);

  /* 2 and 5. Usable prefix reuse per workload and policy. */
  function portion(node, Q, rows, pols = P) {
    policyLegend(node, pols, [CEILING_KEY(), { label: "Range over the seeds",
      color: tok("--ink"), kind: "line" }]);
    groupedBars(node, {
      rows: rows.map((r) => ({ id: r, label: LABEL[r] })),
      series: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p) })),
      value: (r, p) => Q.values[p][r], ref: (r) => Q.ceiling[r],
      xLabel: "Usable prefix reuse: share of prompt tokens served →",
    });
  }
  const portionTable = (Q, rows, pols = P) => [["Workload",
    ...pols.map((p) => POLICY[p]), "No eviction"], rows.map((r) => [LABEL[r],
    ...pols.map((p) => pct((Q.values[p][r] || {}).mean, 1)),
    pct((Q.ceiling[r] || {}).mean, 1)])];
  draw["fig-q2"] = (node) => portion(node, D.q2, D.q2.rows, M);
  tables["fig-q2"] = () => portionTable(D.q2, D.q2.rows, M);

  /* 3. Identification: accuracy per characteristic, and the confusion. */
  draw["fig-q3-acc"] = (node) => {
    const Q = D.q3;
    const series = [
      { id: "accuracy", label: "Requests labelled right",
        color: tok("--seq-4") },
      { id: "balanced", label: "Average of the four workloads' scores",
        color: tok("--seq-2") }];
    legend(node, [...series, { label: "Always guessing " +
      LABEL[Q.majority.label] + " (" + pct(Q.majority.accuracy) + " right; " +
      pct(Q.majority.balanced) + " averaged over workloads)",
      color: tok("--ink"), kind: "dash" }]);
    groupedBars(node, {
      rows: Q.models.map((m) => ({ id: m.id, label: m.label })),
      series, marginLeft: 172, digits: 0,
      seriesRefs: [{ s: series[0].label, v: Q.majority.accuracy },
                   { s: series[1].label, v: Q.majority.balanced }],
      value: (r, s) => ({ mean: Q.models.find((m) => m.id === r)[s] }),
      xLabel: "Share labelled right on runs it never saw (seeds 2–4) →",
    });
  };
  tables["fig-q3-acc"] = () => [["Signals used", "Requests right",
    "Average of the four workloads", "Prompt tokens right"],
    [["Always guess " + LABEL[D.q3.majority.label],
      pct(D.q3.majority.accuracy, 1), pct(D.q3.majority.balanced, 1), "–"],
     ...D.q3.models.map((m) => [m.label, pct(m.accuracy, 1),
       pct(m.balanced, 1), pct(m.tokens, 1)])]];

  draw["fig-q3-cm"] = (node) => {
    const C = D.q3.confusion;
    const labels = C.classes.map((c) => LABEL[c]);
    cells(node, {
      rowLabels: labels, colLabels: labels, values: C.share,
      xLabel: "Labelled as →", yLabel: "Really from ↓",
      title: (i, j) => num(C.counts[i][j]) + " of " + labels[i] +
        "'s requests labelled " + labels[j] + " (" + pct(C.share[i][j], 1) +
        ")",
    });
  };
  tables["fig-q3-cm"] = () => {
    const C = D.q3.confusion;
    return [["Really from ↓ / labelled as →",
      ...C.classes.map((c) => LABEL[c]), "Requests"],
      C.classes.map((c, i) => [LABEL[c], ...C.counts[i].map(num),
        num(C.counts[i].reduce((a, b) => a + b, 0))])];
  };

  /* 4. Capacity: usable reuse against cache size (cache replay lines),
     with the engine runs as hollow markers. */
  draw["fig-q4"] = (node) => {
    const C = D.q4;
    const pols = C.policies;
    legend(node, [...pols.map((p) => ({ label: POLICY[p], color: pol(p) })),
      { label: "No eviction", color: tok("--ink"), kind: "dash" },
      { label: "Measured engine run", color: tok("--ink-2"), hollow: true,
        kind: "dot" },
      { label: "Cache replay of the same seeds", glyph: "×" }]);
    const engine = C.engine.map((e) => ({ ...e, color: pol(e.policy) }));
    // Narrow: only the doubling sizes, so labels do not run together.
    const ticks = width(node) < NARROW ? C.capacities.filter((c, i) =>
      i === C.capacities.length - 1 || Math.log2(c / C.capacities[0]) % 1 === 0)
      : C.capacities;
    lines(node, {
      x: C.capacities, xType: "log", xTicks: ticks, xTick: tokens,
      xDomain: [C.capacities[0] / 1.12, C.capacities.at(-1) * 1.12],
      xFmt: (v) => tokens(v) + " tokens of cache",
      xLabel: "KV-cache capacity in tokens (log scale) →",
      yDomain: C.yDomain, yLabel: "↑ Usable prefix reuse",
      height: 340, marginRight: 64, endLabels: true,
      hrules: [{ y: C.ceiling, label: "No eviction " + pct(C.ceiling, 1) }],
      vrules: [{ x: C.fixed, label: "Cache used in points 1, 2, 5, 6" }],
      series: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p),
        dash: p === "belady" ? "5,3" : null, y: C.curve[p],
        labelDy: C.labelDy[p] || 0 })),
      extra: [
        Plot.dot(engine, { x: "capacity", y: "engine", r: 7,
          stroke: "color", strokeWidth: 2, fill: "none",
          symbol: (d) => SHAPE[d.policy] }),
        Plot.dot(engine, { x: "capacity", y: "replay", r: 4, dx: 13,
          stroke: "color", strokeWidth: 1.5, symbol: "times" }),
        Plot.tip(engine, Plot.pointer({ x: "capacity", y: "engine",
          title: (d) => POLICY[d.policy] + " at " + tokens(d.capacity) +
            " (seed" + (d.seeds.length > 1 ? "s " : " ") + d.seeds.join(", ") +
            ")\nengine " + pct(d.engine, 1) + ", cache replay " +
            pct(d.replay, 1) })),
      ],
    });
  };
  tables["fig-q4"] = () => [["Cache (tokens)", ...D.q4.policies.map(
    (p) => POLICY[p]), "Measured engine runs"], D.q4.capacities.map((c, i) =>
    [tokens(c), ...D.q4.policies.map((p) => pct(D.q4.curve[p][i], 1)),
     D.q4.engine.filter((e) => e.capacity === c).map((e) =>
       POLICY[e.policy] + " " + pct(e.engine, 1) + " (replay " +
       pct(e.replay, 1) + ", seed" + (e.seeds.length > 1 ? "s " : " ") +
       e.seeds.join(", ") + ")").join("; ") || "–"])];

  /* 5. Each workload alone. */
  draw["fig-q5-portion"] = (node) => portion(node, D.q5, D.groups);
  tables["fig-q5-portion"] = () => portionTable(D.q5, D.groups);

  draw["fig-q5-interval"] = (node) => {
    policyLegend(node, P);
    const box = el("div", { class: "multiples two" });
    node.append(box);
    // Lay out every panel first, so each chart is drawn at its final width.
    const panels = D.groups.map((g) => {
      const panel = el("div", { class: "panel" });
      panel.append(el("div", { class: "panel-title" }, LABEL[g] + " alone"));
      const plotNode = el("div", { class: "plot" });
      panel.append(plotNode); box.append(panel);
      return [g, plotNode];
    });
    for (const [g, plotNode] of panels)
      intervalChart(plotNode, D.q5.curves[g], P, 260);
  };
  tables["fig-q5-interval"] = () => {
    const t = D.groups.map((g) => intervalTable(D.q5.curves[g], LABEL[g]));
    return [t[0][0], t.flatMap((x) => x[1])];
  };

  /* 6. Alone against sharing the cache: one line per workload and policy,
     from the hollow mark (alone) to the filled one (in the mix). */
  draw["fig-q6"] = (node) => {
    const Q = D.q6;
    legend(node, [...P.map((p) => ({ label: POLICY[p], color: pol(p) })),
      { label: "Alone", color: tok("--ink-2"), hollow: true, kind: "dot" },
      { label: "Sharing the cache", color: tok("--ink-2"), kind: "dot" },
      { label: "No eviction", color: tok("--ink"), kind: "dash" }]);
    const data = Q.rows.map((r) => ({ ...r, row: LABEL[r.group],
      s: POLICY[r.policy], color: pol(r.policy) }));
    const refs = D.groups.map((g) => ({ row: LABEL[g], v: Q.ceiling[g] }));
    node.append(Plot.plot({
      width: width(node), height: D.groups.length * 86 + 44,
      marginLeft: 108, marginRight: 96, marginBottom: 36, marginTop: 6,
      style: baseStyle(),
      fy: { domain: D.groups.map((g) => LABEL[g]), label: null,
            axis: "left", tickSize: 0, padding: 0.16 },
      y: { domain: P.map((p) => POLICY[p]), axis: null, padding: 0.3 },
      x: { domain: [0, 1], grid: true, tickFormat: (v) => pct(v),
           label: "Usable prefix reuse →" },
      marks: [
        Plot.tickX(refs, { fy: "row", x: "v", stroke: tok("--ink"),
          strokeWidth: 2, strokeDasharray: "4,3" }),
        Plot.link(data, { fy: "row", y1: "s", y2: "s", x1: "alone",
          x2: "mixed", stroke: "color", strokeWidth: 2.5,
          strokeOpacity: 0.75 }),
        Plot.dot(data, { fy: "row", y: "s", x: "alone", r: 5.5,
          stroke: "color", strokeWidth: 2, fill: tok("--surface"),
          symbol: (d) => SHAPE[d.policy] }),
        Plot.dot(data, { fy: "row", y: "s", x: "mixed", r: 5,
          fill: "color", stroke: tok("--surface"), strokeWidth: 1.5,
          symbol: (d) => SHAPE[d.policy] }),
        Plot.text(data, { fy: "row", y: "s", x: 1, dx: 92,
          textAnchor: "end", fill: tok("--ink"), fontSize: 11,
          text: (d) => POLICY[d.policy] + " " + pts(d.mixed - d.alone) }),
        Plot.tip(data, Plot.pointer({ fy: "row", y: "s", x: "mixed",
          title: (d) => d.row + " · " + d.s + "\nalone " + pct(d.alone, 1) +
            ", sharing " + pct(d.mixed, 1) + " (" + pts(d.mixed - d.alone) +
            ")" })),
      ],
    }));
  };
  tables["fig-q6"] = () => [["Workload", "Policy", "Alone",
    "Sharing the cache", "Change", "All 3 seeds agree"],
    D.q6.rows.map((r) => [LABEL[r.group], POLICY[r.policy], pct(r.alone, 1),
      pct(r.mixed, 1), pts(r.mixed - r.alone), r.agree])];

  /* 7. The new policies: each workload's usable reuse against cache size
     (cache replay), one panel per workload. */
  draw["fig-q7-workloads"] = (node) => {
    const Q = D.q7;
    policyLegend(node, Q.policies, [{ label: "No eviction",
      color: tok("--ink"), kind: "dash" }]);
    const box = el("div", { class: "multiples two" });
    node.append(box);
    // Lay out every panel first, so each chart is drawn at its final width.
    const panels = D.groups.map((g) => {
      const panel = el("div", { class: "panel" });
      panel.append(el("div", { class: "panel-title" }, LABEL[g] +
        " in the mix"));
      const plotNode = el("div", { class: "plot" });
      panel.append(plotNode); box.append(panel);
      return [g, plotNode];
    });
    const caps = Q.capacities;
    for (const [g, plotNode] of panels) {
      const C = Q.panels[g];
      // Panels are narrow: label the doubling sizes only.
      const ticks = caps.filter((c, i) => i === caps.length - 1 ||
        Math.log2(c / caps[0]) % 1 === 0);
      lines(plotNode, {
        x: caps, xType: "log", xTicks: ticks, xTick: tokens,
        xDomain: [caps[0] / 1.12, caps.at(-1) * 1.12],
        xFmt: (v) => LABEL[g] + " · " + tokens(v) + " tokens of cache",
        xLabel: "Cache (tokens, log scale) →", yDomain: C.yDomain,
        yLabel: "↑ Usable prefix reuse", height: 250,
        hrules: [{ y: C.ceiling, label: "No eviction " + pct(C.ceiling, 0) }],
        series: Q.policies.map((p) => ({ id: p, label: POLICY[p],
          color: pol(p), y: C.curve[p] })),
      });
    }
  };
  tables["fig-q7-workloads"] = () => [["Workload", "Cache (tokens)",
    ...D.q7.policies.map((p) => POLICY[p]), "No eviction"],
    D.groups.flatMap((g) => D.q7.capacities.map((c, i) => [LABEL[g],
      tokens(c), ...D.q7.policies.map((p) =>
        pct(D.q7.panels[g].curve[p][i], 1)),
      pct(D.q7.panels[g].ceiling, 1)]))];

  /* 7. Evicted blocks per second, by the workload that last used them. */
  const perSec = (v) => (v == null || isNaN(v)) ? "–" :
    v.toFixed(v < 10 ? 1 : 0) + "/s";
  draw["fig-q7-evictions"] = (node) => {
    const E = D.q7.evictions;
    const pols = M.filter((p) => E[p]);
    policyLegend(node, pols, [{ label: "Range over the seeds",
      color: tok("--ink"), kind: "line" }]);
    const rows = [...D.groups, "all"];
    const top = Math.max(...pols.flatMap((p) => rows.map((r) =>
      E[p][r].max)));
    groupedBars(node, {
      rows: rows.map((r) => ({ id: r, label: LABEL[r] })),
      series: pols.map((p) => ({ id: p, label: POLICY[p], color: pol(p) })),
      value: (r, p) => E[p][r], xDomain: [0, top * 1.15], fmt: perSec,
      tickFormat: (v) => v.toLocaleString("en-US"),
      xLabel: "Blocks evicted per second, by the workload that last used " +
        "them →",
    });
  };
  tables["fig-q7-evictions"] = () => {
    const E = D.q7.evictions;
    const pols = M.filter((p) => E[p]);
    return [["Workload", ...pols.flatMap((p) => [POLICY[p] + " evicted/s",
      POLICY[p] + ": needed again"])], [...D.groups, "all"].map((r) =>
      [LABEL[r], ...pols.flatMap((p) => [perSec(E[p][r].mean),
        pct(E[p][r].regretted, 0)])])];
  };

  function renderAll() {
    for (const [id, fn] of Object.entries(draw)) {
      const node = document.getElementById(id);
      if (!node) continue;
      node.replaceChildren();
      const plotNode = el("div", { class: "plot" });
      node.append(plotNode);
      try { fn(plotNode); } catch (err) {
        plotNode.append(el("p", {}, "This chart could not be drawn: " + err));
      }
      try {
        const t = tables[id] && tables[id]();
        if (t) tableView(node, t[0], t[1]);
      } catch (err) {
        node.append(el("p", {}, "Its table could not be built: " + err));
      }
    }
  }

  let pending = null;
  const later = () => { clearTimeout(pending);
    pending = setTimeout(renderAll, 120); };
  renderAll();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", later);
  new MutationObserver(later).observe(root, { attributes: true,
    attributeFilter: ["data-theme"] });
  let lastWidth = document.body.clientWidth;
  new ResizeObserver(() => { const w = document.body.clientWidth;
    if (Math.abs(w - lastWidth) > 40) { lastWidth = w; later(); } })
    .observe(document.body);
})();
