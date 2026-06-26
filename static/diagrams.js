window.DIAG = {};

function _dlang() {
  return window.S && S.lang ? S.lang : "th";
}
function _dt(th, en) {
  return _dlang() === "th" ? th : en;
}
function _dsvg(container, w, h) {
  d3.select(container).selectAll("*").remove();
  return d3
    .select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${w} ${h}`)
    .attr("width", "100%")
    .style("max-width", w + "px")
    .style("background", "#0b0e13")
    .style("border", "1px solid var(--line)")
    .style("border-radius", "8px");
}
function _motion(sel, dur, path, begin) {
  const a = sel.append("animateMotion").attr("dur", dur).attr("repeatCount", "indefinite").attr("path", path);
  if (begin) a.attr("begin", begin);
  return a;
}
function _fade(sel, values, dur, begin) {
  const a = sel
    .append("animate")
    .attr("attributeName", "opacity")
    .attr("values", values)
    .attr("dur", dur)
    .attr("repeatCount", "indefinite");
  if (begin) a.attr("begin", begin);
  return a;
}

window.DIAG.lhcb_tour = function (c) {
  const W = 760, H = 220, midY = 100;
  const svg = _dsvg(c, W, H);
  const parts = [
    ["VELO", "VELO"], ["tracker", "tracker"], [_dt("แม่เหล็ก", "magnet"), "magnet"],
    ["RICH", "RICH"], ["ECAL", "ECAL"], ["HCAL", "HCAL"], ["muon", "muon"],
  ];
  const x0 = 70, gap = (W - x0 - 20) / parts.length;
  svg.append("circle").attr("cx", 32).attr("cy", midY).attr("r", 4).attr("fill", "#fff");
  svg.append("text").attr("x", 18).attr("y", midY - 8).attr("fill", "#fff").attr("font-size", 11).text("pp");
  parts.forEach((p, i) => {
    const x = x0 + i * gap;
    const isMag = p[1] === "magnet", isE = p[1] === "ECAL";
    svg.append("rect").attr("x", x).attr("y", 45).attr("width", gap - 10).attr("height", 110)
      .attr("fill", isMag ? "#243b6b" : isE ? "#3a2a17" : "#1c2230").attr("stroke", "var(--line)").attr("rx", 4);
    svg.append("text").attr("x", x + (gap - 10) / 2).attr("y", 172).attr("text-anchor", "middle")
      .attr("fill", isE ? "var(--accent2)" : "var(--muted)").attr("font-size", 10).text(p[0]);
  });
  const eX = x0 + 4 * gap + (gap - 10) / 2;
  const magX = x0 + 2 * gap + (gap - 10) / 2;
  const ph = svg.append("circle").attr("r", 5).attr("fill", "var(--seed)");
  _motion(ph, "3s", `M32,${midY} L${eX},${midY}`);
  const ch = svg.append("circle").attr("r", 4).attr("fill", "var(--accent)");
  _motion(ch, "3s", `M32,${midY} L${magX},${midY} Q${magX + 50},${midY} ${W - 25},${midY - 60}`);
  svg.append("circle").attr("cx", x0).attr("cy", 22).attr("r", 4).attr("fill", "var(--seed)");
  svg.append("text").attr("x", x0 + 9).attr("y", 26).attr("fill", "var(--muted)").attr("font-size", 10)
    .text(_dt("โฟตอน (กลาง → ตรง, หยุดที่ ECAL)", "photon (neutral → straight, stops at ECAL)"));
  svg.append("circle").attr("cx", x0 + 300).attr("cy", 22).attr("r", 4).attr("fill", "var(--accent)");
  svg.append("text").attr("x", x0 + 309).attr("y", 26).attr("fill", "var(--muted)").attr("font-size", 10)
    .text(_dt("มีประจุ (โค้งในแม่เหล็ก)", "charged (bends in magnet)"));
};

window.DIAG.coordinates = function (c) {
  const W = 760, H = 250;
  const wrap = d3.select(c);
  wrap.selectAll("*").remove();
  const svg = wrap.append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%")
    .style("max-width", W + "px").style("background", "#0b0e13").style("border", "1px solid var(--line)")
    .style("border-radius", "8px");
  const ox = 90, oy = 150;
  svg.append("line").attr("x1", ox).attr("y1", oy).attr("x2", W - 40).attr("y2", oy)
    .attr("stroke", "var(--muted)").attr("stroke-width", 1.5).attr("marker-end", "url(#ar)");
  svg.append("defs").append("marker").attr("id", "ar").attr("markerWidth", 8).attr("markerHeight", 8)
    .attr("refX", 6).attr("refY", 3).attr("orient", "auto").append("path").attr("d", "M0,0 L6,3 L0,6 Z").attr("fill", "var(--muted)");
  svg.append("text").attr("x", W - 36).attr("y", oy + 4).attr("fill", "var(--muted)").attr("font-size", 12).text("z (beam)");
  svg.append("circle").attr("cx", ox).attr("cy", oy).attr("r", 3).attr("fill", "#fff");
  const ray = svg.append("line").attr("x1", ox).attr("y1", oy).attr("stroke", "var(--seed)").attr("stroke-width", 2.5);
  const dot = svg.append("circle").attr("r", 5).attr("fill", "var(--seed)");
  const lbl = svg.append("text").attr("x", ox).attr("y", 40).attr("fill", "var(--fg)").attr("font-size", 13);
  const lbl2 = svg.append("text").attr("x", ox).attr("y", 62).attr("fill", "var(--muted)").attr("font-size", 12);
  const slider = wrap.append("input").attr("type", "range").attr("min", 4).attr("max", 90).attr("value", 25)
    .style("width", "60%").style("margin-top", "8px").style("display", "block");
  function draw(thetaDeg) {
    const th = (thetaDeg * Math.PI) / 180;
    const L = 230;
    const ex = ox + L * Math.cos(th), ey = oy - L * Math.sin(th);
    ray.attr("x2", ex).attr("y2", ey);
    dot.attr("cx", ex).attr("cy", ey);
    const eta = -Math.log(Math.tan(th / 2));
    lbl.text(`θ = ${thetaDeg}°`);
    lbl2.text(`η = -ln tan(θ/2) = ${eta.toFixed(2)}  ${eta > 2 ? _dt("(forward — โซน LHCb)", "(forward — LHCb region)") : ""}`);
  }
  slider.on("input", function () { draw(+this.value); });
  draw(25);
};

window.DIAG.where_from = function (c) {
  const W = 760, H = 220, cx = W / 2, cy = 110;
  const svg = _dsvg(c, W, H);
  [-1, 1].forEach((s) => {
    const pr = svg.append("circle").attr("r", 6).attr("fill", "#8fa3bf");
    _motion(pr, "2.4s", `M${cx + s * 340},${cy} L${cx + s * 26},${cy}`);
    _fade(pr, "1;1;0;0;1", "2.4s");
  });
  svg.append("text").attr("x", 24).attr("y", cy - 10).attr("fill", "#8fa3bf").attr("font-size", 12).text("p");
  svg.append("text").attr("x", W - 32).attr("y", cy - 10).attr("fill", "#8fa3bf").attr("font-size", 12).text("p");
  const pi = svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 8).attr("fill", "var(--truth)");
  _fade(pi, "0;0;1;1;0", "2.4s");
  svg.append("text").attr("x", cx + 12).attr("y", cy - 10).attr("text-anchor", "middle").attr("fill", "var(--truth)").attr("font-size", 12).text("π⁰");
  [[-70], [70]].forEach((dy, i) => {
    const g = svg.append("circle").attr("r", 5).attr("fill", "var(--seed)");
    _motion(g, "2.4s", `M${cx},${cy} L${cx + 290},${cy + dy[0]}`);
    _fade(g, "0;0;0;1;1", "2.4s");
    svg.append("text").attr("x", cx + 250).attr("y", cy + dy[0] * 0.92).attr("fill", "var(--seed)").attr("font-size", 12).text("γ");
  });
  svg.append("text").attr("x", cx).attr("y", H - 14).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 10)
    .text(_dt("pp → π⁰ → γγ  (สองโฟตอน)", "pp → π⁰ → γγ  (the two photons)"));
};

window.DIAG.shower = function (c) {
  const W = 760, H = 250;
  const svg = _dsvg(c, W, H);
  svg.append("rect").attr("x", 110).attr("y", 25).attr("width", W - 150).attr("height", H - 70)
    .attr("fill", "#12161d").attr("stroke", "var(--line)");
  svg.append("text").attr("x", 70).attr("y", H / 2 - 12).attr("fill", "var(--seed)").attr("font-size", 13).text("γ");
  const x0 = 110, y0 = H / 2 - 16, dx = 64;
  const segs = [[55, H / 2 - 16, x0, y0]];
  (function branch(x, y, depth, dir) {
    if (depth > 4 || x > W - 55) return;
    const nx = x + dx, ny = y + dir * (46 / depth);
    segs.push([x, y, nx, ny]);
    branch(nx, ny, depth + 1, 1);
    branch(nx, ny, depth + 1, -1);
  })(x0, y0, 1, 1);
  (function branch(x, y, depth, dir) {
    if (depth > 4 || x > W - 55) return;
    const nx = x + dx, ny = y + dir * (46 / depth);
    segs.push([x, y, nx, ny]);
    branch(nx, ny, depth + 1, 1);
    branch(nx, ny, depth + 1, -1);
  })(x0, y0, 1, -1);
  segs.forEach((s, i) => {
    const ln = svg.append("line").attr("x1", s[0]).attr("y1", s[1]).attr("x2", s[2]).attr("y2", s[3])
      .attr("stroke", "var(--accent)").attr("stroke-width", 1.6).attr("opacity", 0);
    _fade(ln, "0;1;1;0", "4s", `${i * 0.07}s`);
  });
  svg.append("text").attr("x", W - 60).attr("y", 40).attr("text-anchor", "end").attr("fill", "var(--muted)").attr("font-size", 10)
    .text(_dt("pair production + bremsstrahlung → cascade", "pair production + bremsstrahlung → cascade"));
};

window.DIAG.sampling = function (c) {
  const W = 760, H = 220;
  const svg = _dsvg(c, W, H);
  const n = 12, x0 = 150, lw = (W - x0 - 130) / n, top = 40, h = 110;
  for (let i = 0; i < n; i++) {
    const x = x0 + i * lw, scint = i % 2 === 1;
    const r = svg.append("rect").attr("x", x).attr("y", top).attr("width", lw - 2).attr("height", h)
      .attr("fill", scint ? "#19c3c3" : "#4a4a4a").attr("opacity", scint ? 0.5 : 1);
    if (scint) _fade(r, "0.3;1;0.3", "2.6s", `${i * 0.13}s`);
  }
  svg.append("text").attr("x", x0).attr("y", top - 12).attr("fill", "#9aa7b4").attr("font-size", 10).text(_dt("■ absorber (W/Pb)", "■ absorber (W/Pb)"));
  svg.append("text").attr("x", x0 + 170).attr("y", top - 12).attr("fill", "#19c3c3").attr("font-size", 10).text(_dt("■ scintillator (เปล่งแสง)", "■ scintillator (emits light)"));
  const ph = svg.append("circle").attr("r", 5).attr("fill", "var(--seed)");
  _motion(ph, "2.6s", `M60,${top + h / 2} L${x0},${top + h / 2}`);
  svg.append("path").attr("d", `M${W - 105},${top + 28} L${W - 55},${top + 8} L${W - 55},${top + h - 8} L${W - 105},${top + h - 28} Z`).attr("fill", "#ff7a3d");
  svg.append("text").attr("x", W - 80).attr("y", top + h + 18).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 10).text("PMT");
};

window.DIAG.readout = function (c) {
  const W = 760, H = 180, y = 80;
  const svg = _dsvg(c, W, H);
  const nodes = [
    [_dt("แสง scintillation", "scintillation light"), 90],
    ["fibre / WLS", 250],
    ["PMT", 400],
    ["ADC / DRS4", 540],
    [_dt("MeV, ns", "MeV, ns"), 690],
  ];
  nodes.forEach((nd, i) => {
    svg.append("circle").attr("cx", nd[1]).attr("cy", y).attr("r", 6).attr("fill", i === nodes.length - 1 ? "var(--good)" : "var(--accent)");
    svg.append("text").attr("x", nd[1]).attr("y", y - 16).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 10).text(nd[0]);
    if (i < nodes.length - 1) {
      svg.append("line").attr("x1", nd[1] + 8).attr("y1", y).attr("x2", nodes[i + 1][1] - 8).attr("y2", y).attr("stroke", "var(--line)");
    }
  });
  const pulse = svg.append("circle").attr("r", 5).attr("fill", "var(--seed)");
  _motion(pulse, "3s", `M90,${y} L690,${y}`);
  svg.append("text").attr("x", W / 2).attr("y", H - 14).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 10)
    .text(_dt("พลังงาน = ประจุที่อินทิเกรต · เวลา = รูปคลื่น", "energy = integrated charge · time = waveform"));
};

window.DIAG.reco_cluster = function (c) {
  const W = 760, H = 250;
  const svg = _dsvg(c, W, H);
  const g = 5, cell = 38, x0 = 230, y0 = 30;
  let energies = [];
  for (let r = 0; r < g; r++) {
    energies[r] = [];
    for (let cc = 0; cc < g; cc++) {
      const d2 = (r - 2) ** 2 + (cc - 2) ** 2;
      energies[r][cc] = Math.exp(-d2 / 2.2);
    }
  }
  const col = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
  for (let r = 0; r < g; r++) {
    for (let cc = 0; cc < g; cc++) {
      svg.append("rect").attr("x", x0 + cc * cell).attr("y", y0 + r * cell).attr("width", cell - 2).attr("height", cell - 2)
        .attr("fill", col(energies[r][cc])).attr("stroke", "#0b0e13");
    }
  }
  const sx = x0 + 2 * cell + cell / 2, sy = y0 + 2 * cell + cell / 2;
  const star = svg.append("text").attr("x", sx).attr("y", sy + 6).attr("text-anchor", "middle").attr("fill", "var(--seed)").attr("font-size", 20).attr("font-weight", 700).text("★");
  _fade(star, "0.4;1;0.4", "2s");
  const box = svg.append("rect").attr("x", x0 + 1 * cell).attr("y", y0 + 1 * cell).attr("width", 3 * cell).attr("height", 3 * cell)
    .attr("fill", "none").attr("stroke", "var(--reco)").attr("stroke-width", 2.5).attr("stroke-dasharray", "5 4").attr("opacity", 0);
  _fade(box, "0;0;1;1", "3s");
  svg.append("text").attr("x", sx).attr("y", sy - 8).attr("text-anchor", "middle").attr("fill", "var(--reco)").attr("font-size", 18).text("+");
  svg.append("text").attr("x", 30).attr("y", y0 + 30).attr("fill", "var(--muted)").attr("font-size", 11).text(_dt("★ seed =", "★ seed ="));
  svg.append("text").attr("x", 30).attr("y", y0 + 48).attr("fill", "var(--muted)").attr("font-size", 11).text(_dt("เซลล์สูงสุด", "max cell"));
  svg.append("text").attr("x", 30).attr("y", y0 + 90).attr("fill", "var(--reco)").attr("font-size", 11).text(_dt("กรอบ = หน้าต่าง", "box = window"));
  svg.append("text").attr("x", 30).attr("y", y0 + 108).attr("fill", "var(--reco)").attr("font-size", 11).text("3×3 / 5×5");
  svg.append("text").attr("x", 30).attr("y", y0 + 150).attr("fill", "var(--reco)").attr("font-size", 11).text(_dt("+ = centroid", "+ = centroid"));
};

window.DIAG.measure_what = function (c) {
  const W = 760, H = 240, m = { l: 64, r: 20, t: 22, b: 46 };
  const svg = _dsvg(c, W, H);
  const res = (e) => Math.sqrt((0.1 / Math.sqrt(e)) ** 2 + 0.01 ** 2);
  const E = d3.range(1, 100.5, 0.5);
  const x = d3.scaleLog().domain([1, 100]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 0.11]).range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5, "~s"));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, "%"));
  const line = d3.line().x((d) => x(d)).y((d) => y(res(d)));
  svg.append("path").attr("d", line(E)).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.5);
  svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 11).text("E [GeV]");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -H / 2).attr("y", 16).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 11).text("σ_E / E");
  svg.append("text").attr("x", W - 30).attr("y", m.t + 14).attr("text-anchor", "end").attr("fill", "var(--accent2)").attr("font-size", 13).text("σ_E/E = 10%/√E ⊕ 1%");
  svg.append("text").attr("x", W - 30).attr("y", m.t + 32).attr("text-anchor", "end").attr("fill", "var(--muted)").attr("font-size", 10).text(_dt("ยิ่ง E สูง ยิ่งแม่น", "higher E → better resolution"));
};
