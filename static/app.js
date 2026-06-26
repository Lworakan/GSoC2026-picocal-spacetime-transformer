const S = {
  lang: "th",
  file: null,
  event: null,
  geometry: null,
  dict: null,
  explainers: null,
  view: "fundamentals",
  energyMode: "energy",
  colorLog: true,
};

const UI = {
  title: { th: "PicoCal Data Explorer", en: "PicoCal Data Explorer" },
  file: { th: "ไฟล์", en: "File" },
  tab_fundamentals: { th: "พื้นฐาน", en: "Fundamentals" },
  fund_title: { th: "พื้นฐาน: จากการชนถึงการวัด", en: "Fundamentals: from collision to measurement" },
  fund_sub: {
    th: "อิงจาก paper จริง (The LHCb PicoCal, NIMA 1079 (2025) 170608) + ฟิสิกส์มาตรฐาน",
    en: "Grounded in the paper (The LHCb PicoCal, NIMA 1079 (2025) 170608) + standard physics",
  },
  tab_overview: { th: "ภาพรวม", en: "Overview" },
  tab_sensor: { th: "เซนเซอร์", en: "Detector" },
  tab_threed: { th: "อีเวนต์ 3D", en: "3D Event" },
  tv_hint: { th: "ลากเพื่อหมุน · สกรอลล์ซูม · เปิด idealize เพื่อเทียบ shower ในอุดมคติ", en: "drag to orbit · scroll to zoom · toggle idealize to compare a textbook shower" },
  tab_shower: { th: "ความลึก shower", en: "Shower depth" },
  tab_dictionary: { th: "พจนานุกรมข้อมูล", en: "Data dictionary" },
  tab_tour: { th: "ทัวร์นำชม", en: "Guided tour" },
  tab_exploration: { th: "สำรวจ & สูตร", en: "Exploration & Formulas" },
  source: {
    th: "ข้อมูล: Geant4 simulation ที่ได้รับมา — ทรี clusters_matched",
    en: "Data: provided Geant4 simulation — tree clusters_matched",
  },
  loading: { th: "กำลังโหลด…", en: "Loading…" },
  click_point: { th: "คลิกจุดเพื่อเปิด event นั้นในหน้าเซนเซอร์", en: "Click a point to open that event in the detector view" },
  truth_entry: { th: "จุดเข้าจริงของโฟตอน", en: "Truth photon entry" },
  reco_cluster: { th: "ตำแหน่ง cluster ที่ reco", en: "Reconstructed cluster" },
  seed_cell: { th: "เซลล์ seed (พลังงานสูงสุด)", en: "Seed cell (max energy)" },
  full_face: { th: "หน้า ECAL ทั้งหมด (ตำแหน่งจริงของหน้าต่าง)", en: "Full ECAL face (real window location)" },
  window_zoom: { th: "หน้าต่างโมดูล 5×5 (ซูม)", en: "5×5 module window (zoom)" },
  front: { th: "หน้า", en: "front" },
  back: { th: "หลัง", en: "back" },
  total: { th: "รวม", en: "total" },
  logc: { th: "สีแบบ log", en: "log color" },
  event_label: { th: "event", en: "event" },
  energy_gev: { th: "พลังงานจริง (GeV)", en: "truth energy (GeV)" },
  n_cells: { th: "จำนวนเซลล์ต่อ cluster (= sequence length)", en: "cells per cluster (= sequence length)" },
  response_x: { th: "ΔE/E", en: "ΔE/E" },
  dr_x: { th: "ระยะ cluster ↔ truth (mm)", en: "cluster ↔ truth distance (mm)" },
  counts: { th: "จำนวน", en: "count" },
  lin: { th: "เชิงเส้น", en: "linear" },
  log: { th: "log", en: "log" },
  pick_event: { th: "เลือก event:", en: "pick event:" },
  search: { th: "ค้นหา…", en: "search…" },
  col_name: { th: "ชื่อ", en: "name" },
  col_level: { th: "ระดับ", en: "level" },
  col_unit: { th: "หน่วย", en: "unit" },
  col_present: { th: "ที่มา", en: "source" },
  col_what: { th: "คืออะไร", en: "what" },
  col_note: { th: "หมายเหตุ", en: "note" },
  col_range: { th: "ช่วงค่า (ไฟล์นี้)", en: "range (this file)" },
  derived: { th: "คำนวณ", en: "derived" },
  why_eda_title: { th: "ทำไมต้องสำรวจข้อมูล ทั้งที่มี guideline แล้ว?", en: "Why explore the data when we already have a guideline?" },
};

function t(o) {
  if (o == null) return "";
  if (typeof o === "string") return o;
  return o[S.lang] != null ? o[S.lang] : o.en;
}
function L(key) { return t(UI[key]); }

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}
function fmt(v, d = 2) {
  if (v == null || !isFinite(v)) return "—";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-2)) return v.toExponential(2);
  return (+v).toFixed(d);
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function mathify(root) {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
      });
    } catch (e) {}
  }
}

function applyStaticI18n() {
  document.documentElement.lang = S.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = L(el.getAttribute("data-i18n"));
  });
}

function tip(html, ev) {
  const tt = document.getElementById("tooltip");
  if (html == null) { tt.hidden = true; return; }
  tt.hidden = false;
  tt.innerHTML = html;
  const x = ev.clientX + 14, y = ev.clientY + 14;
  tt.style.left = Math.min(x, window.innerWidth - tt.offsetWidth - 8) + "px";
  tt.style.top = Math.min(y, window.innerHeight - tt.offsetHeight - 8) + "px";
}

function explainPanel(item) {
  const div = document.createElement("div");
  div.className = "explain";
  div.innerHTML =
    `<div><span class="lab">${t(item.title)}</span></div>` +
    (item.formula_latex ? `<div class="formula">\\(${item.formula_latex}\\)</div>` : "") +
    `<div><span class="lab">?</span>${t(item.what)}</div>` +
    (item.why ? `<div><span class="lab">→</span>${t(item.why)}</div>` : "") +
    (item.ml_analogy ? `<div class="muted">ML: ${t(item.ml_analogy)}</div>` : "");
  return div;
}
function findEx(id) {
  const e = S.explainers;
  return (e.plots.concat(e.formulas)).find((x) => x.id === id);
}

function setView(name) {
  S.view = name;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  render();
}

function render() {
  if (window.ThreeView) window.ThreeView.unmount();
  if (!S.file) return;
  const fn = {
    fundamentals: renderFundamentals, overview: renderOverview, sensor: renderSensor,
    threed: renderThreeD, shower: renderShower, dictionary: renderDictionary,
    tour: renderTour, exploration: renderExploration,
  }[S.view];
  fn();
}

function viewEl() { return document.getElementById("view-" + S.view); }
function loadingInto(el) { el.innerHTML = `<div class="loading">${L("loading")}</div>`; }

async function renderFundamentals() {
  const el = document.getElementById("view-fundamentals");
  loadingInto(el);
  if (!S.explainers) S.explainers = await api(`/api/explainers`);
  clear(el);
  const intro = document.createElement("div");
  intro.className = "card";
  intro.innerHTML = `<h3>${L("fund_title")}</h3><div class="muted">${L("fund_sub")}</div>`;
  el.appendChild(intro);
  (S.explainers.fundamentals || []).forEach((f, i) => {
    const c = document.createElement("div");
    c.className = "card";
    const title = t(f.title).replace(/^\d+\.\s*/, "");
    c.innerHTML = `<h3>${i + 1}. ${title}</h3><div class="fundbody">${t(f.body)}</div>`;
    el.appendChild(c);
    if (window.DIAG && window.DIAG[f.id]) {
      const fig = document.createElement("div");
      fig.className = "diagram";
      c.appendChild(fig);
      try {
        window.DIAG[f.id](fig);
      } catch (e) {}
    }
    mathify(c);
  });
}

async function renderOverview() {
  const el = document.getElementById("view-overview");
  loadingInto(el);
  const ov = await api(`/api/files/${S.file}/overview`);
  clear(el);
  const head = document.createElement("div");
  head.className = "card";
  head.innerHTML = `<h3>${S.file}</h3><div class="muted">${ov.entries.length} entries · ${ov.n_events} events · ${L("click_point")}</div>`;
  el.appendChild(head);
  const card = document.createElement("div");
  card.className = "card";
  el.appendChild(card);

  const W = Math.min(960, el.clientWidth - 40), H = 460, m = { l: 60, r: 20, t: 16, b: 44 };
  const svg = d3.select(card).append("svg").attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain(d3.extent(ov.entries, (d) => d.event)).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(ov.entries, (d) => d.sig_flux_eTot)]).nice().range([H - m.b, m.t]);
  const maxMul = d3.max(ov.entries, (d) => d.multiplicity) || 1;
  const col = d3.scaleSequential(d3.interpolateTurbo).domain([1, maxMul]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y));
  svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 12).text(L("event_label"));
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -H / 2).attr("y", 16).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 12).text(L("energy_gev"));
  svg.append("g").selectAll("circle").data(ov.entries).join("circle")
    .attr("cx", (d) => x(d.event)).attr("cy", (d) => y(d.sig_flux_eTot)).attr("r", 4)
    .attr("fill", (d) => col(d.multiplicity)).attr("opacity", 0.7).style("cursor", "pointer")
    .on("mousemove", (ev, d) => tip(`<b>event ${d.event}</b><br>entry ${d.tree_entry}<br>E=${fmt(d.sig_flux_eTot)} GeV<br>dr=${fmt(d.sig_dr_matched)} mm<br>mult=${d.multiplicity}`, ev))
    .on("mouseleave", () => tip(null))
    .on("click", (ev, d) => { S.event = d.event; setView("sensor"); });
}

async function renderThreeD() {
  const el = document.getElementById("view-threed");
  loadingInto(el);
  const ov = await api(`/api/files/${S.file}/overview`);
  if (S.event == null) S.event = ov.entries[0].event;
  const d = await api(`/api/files/${S.file}/event/${S.event}`);
  const uniq = Array.from(new Set(ov.entries.map((e) => e.event))).sort((a, b) => a - b);
  clear(el);
  const bar = document.createElement("div");
  bar.className = "toolbar card";
  bar.innerHTML = `<label>${L("pick_event")} <select id="tv-ev"></select></label><span class="muted">${L("tv_hint")}</span>`;
  el.appendChild(bar);
  const sel = bar.querySelector("#tv-ev");
  uniq.forEach((e) => { const o = document.createElement("option"); o.value = e; o.textContent = e; if (e === S.event) o.selected = true; sel.appendChild(o); });
  sel.onchange = () => { S.event = +sel.value; renderThreeD(); };
  const holder = document.createElement("div");
  holder.className = "card";
  holder.style.padding = "0";
  holder.style.overflow = "hidden";
  const stage = document.createElement("div");
  stage.className = "tv-stage";
  holder.appendChild(stage);
  el.appendChild(holder);
  if (window.ThreeView && window.THREE) {
    window.ThreeView.mount(stage, d, S.lang);
  } else {
    stage.innerHTML = `<div class="loading">three.js not loaded</div>`;
  }
}

function energyOf(c) { return S.energyMode === "front" ? c.front : S.energyMode === "back" ? c.back : c.energy; }

async function renderSensor() {
  const el = document.getElementById("view-sensor");
  loadingInto(el);
  if (S.event == null) {
    const ov = await api(`/api/files/${S.file}/overview`);
    S.event = ov.entries[0].event;
  }
  const d = await api(`/api/files/${S.file}/event/${S.event}`);
  if (!S.geometry) S.geometry = await api(`/api/geometry`);
  clear(el);

  const bar = document.createElement("div");
  bar.className = "toolbar card";
  const events = (await api(`/api/files/${S.file}/overview`)).entries.map((e) => e.event);
  const uniq = Array.from(new Set(events)).sort((a, b) => a - b);
  bar.innerHTML = `<label>${L("pick_event")} <select id="ev-sel"></select></label>`;
  ["energy", "front", "back"].forEach((mode) => {
    const b = document.createElement("button");
    b.className = "pill" + (S.energyMode === mode ? " active" : "");
    b.textContent = mode === "energy" ? L("total") : mode === "front" ? L("front") : L("back");
    b.onclick = () => { S.energyMode = mode; renderSensor(); };
    bar.appendChild(b);
  });
  const lg = document.createElement("button");
  lg.className = "pill" + (S.colorLog ? " active" : "");
  lg.textContent = L("logc");
  lg.onclick = () => { S.colorLog = !S.colorLog; renderSensor(); };
  bar.appendChild(lg);
  el.appendChild(bar);
  const sel = bar.querySelector("#ev-sel");
  uniq.forEach((e) => { const o = document.createElement("option"); o.value = e; o.textContent = e; if (e === S.event) o.selected = true; sel.appendChild(o); });
  sel.onchange = () => { S.event = +sel.value; renderSensor(); };

  const row = document.createElement("div");
  row.className = "row";
  el.appendChild(row);
  const faceCard = document.createElement("div");
  faceCard.className = "card grow";
  faceCard.innerHTML = `<h3>${L("full_face")}</h3>`;
  const zoomCard = document.createElement("div");
  zoomCard.className = "card grow";
  zoomCard.innerHTML = `<h3>${L("window_zoom")} — event ${d.event}</h3>`;
  row.appendChild(faceCard);
  row.appendChild(zoomCard);

  drawFace(faceCard, d);
  drawWindow(zoomCard, d);

  const legend = document.createElement("div");
  legend.className = "card";
  legend.innerHTML =
    `<span class="legend-mark" style="background:var(--truth)"></span>${L("truth_entry")} &nbsp;&nbsp;` +
    `<span class="legend-mark" style="background:var(--reco)"></span>${L("reco_cluster")} &nbsp;&nbsp;` +
    `<span class="legend-mark" style="background:var(--seed)"></span>${L("seed_cell")}`;
  el.appendChild(legend);
}

function drawFace(card, d) {
  const W = Math.max(320, Math.min(520, card.clientWidth - 28)), H = 460, m = 30;
  const g = S.geometry;
  const xs = g.map((mm) => mm.x).filter((v) => v != null);
  const ys = g.map((mm) => mm.y).filter((v) => v != null);
  const ext = Math.max(d3.max(xs.map(Math.abs)), d3.max(ys.map(Math.abs)));
  const x = d3.scaleLinear().domain([-ext, ext]).range([m, W - m]);
  const y = d3.scaleLinear().domain([-ext, ext]).range([H - m, m]);
  const svg = d3.select(card).append("svg").attr("width", W).attr("height", H);
  const win = new Set(d.window_modules.map((w) => w.imodx + "," + w.jmody));
  svg.append("g").selectAll("rect").data(g).join("rect")
    .attr("x", (mm) => x(mm.x) - 1.5).attr("y", (mm) => y(mm.y) - 1.5)
    .attr("width", 3).attr("height", 3)
    .attr("fill", (mm) => (win.has(mm.imodx + "," + mm.jmody) ? "var(--accent2)" : "#2c3banti"))
    .attr("fill", (mm) => (win.has(mm.imodx + "," + mm.jmody) ? "var(--accent2)" : "#33405a"))
    .attr("opacity", (mm) => (win.has(mm.imodx + "," + mm.jmody) ? 1 : 0.5));
  d.truth_photons.forEach((p) => {
    svg.append("text").attr("x", x(p.entry_x)).attr("y", y(p.entry_y) + 4).attr("text-anchor", "middle").attr("fill", "var(--truth)").attr("font-size", 14).text("×");
  });
  svg.append("text").attr("x", W / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 11).text("x [mm]");
}

function drawWindow(card, d) {
  const cells = [];
  d.clusters.forEach((c, ci) => c.cells.forEach((cell) => cells.push(Object.assign({ ci }, cell))));
  if (!cells.length) { card.innerHTML += `<div class="muted">no cells</div>`; return; }
  const W = Math.max(320, Math.min(520, card.clientWidth - 28)), H = 460, m = 36;
  const pad = 30;
  let xsAll = cells.map((c) => c.x), ysAll = cells.map((c) => c.y);
  d.truth_photons.forEach((p) => { xsAll.push(p.entry_x); ysAll.push(p.entry_y); });
  d.clusters.forEach((c) => { xsAll.push(c.x_cluster); ysAll.push(c.y_cluster); });
  const xe = d3.extent(xsAll), ye = d3.extent(ysAll);
  const spanx = (xe[1] - xe[0]) || 100, spany = (ye[1] - ye[0]) || 100;
  const span = Math.max(spanx, spany) + pad;
  const cx = (xe[0] + xe[1]) / 2, cy = (ye[0] + ye[1]) / 2;
  const x = d3.scaleLinear().domain([cx - span / 2, cx + span / 2]).range([m, W - m]);
  const y = d3.scaleLinear().domain([cy - span / 2, cy + span / 2]).range([H - m, m]);
  const scale = (W - 2 * m) / span;

  const evals = cells.map(energyOf).filter((v) => v > 0);
  const emax = d3.max(cells, energyOf) || 1;
  const emin = d3.min(evals) || 1;
  const color = S.colorLog
    ? d3.scaleSequentialLog(d3.interpolateViridis).domain([Math.max(emin, emax / 1e4), emax])
    : d3.scaleSequential(d3.interpolateViridis).domain([0, emax]);

  const svg = d3.select(card).append("svg").attr("width", W).attr("height", H);
  svg.append("g").selectAll("rect").data(cells).join("rect")
    .attr("x", (c) => x(c.x) - ((c.pitch_derived || 6) * scale) / 2)
    .attr("y", (c) => y(c.y) - ((c.pitch_derived || 6) * scale) / 2)
    .attr("width", (c) => Math.max(2, (c.pitch_derived || 6) * scale * 0.92))
    .attr("height", (c) => Math.max(2, (c.pitch_derived || 6) * scale * 0.92))
    .attr("fill", (c) => (energyOf(c) > 0 ? color(energyOf(c)) : "#11161f"))
    .attr("stroke", "#0b0e13").attr("stroke-width", 0.5)
    .style("cursor", "pointer")
    .on("mousemove", (ev, c) => tip(
      `<b>cell</b> mod(${c.imodx},${c.jmody}) icell ${c.icell}<br>` +
      `x=${fmt(c.x)} y=${fmt(c.y)} mm<br>` +
      `E=${fmt(c.energy)} (front ${fmt(c.front)} / back ${fmt(c.back)}) MeV<br>` +
      `pitch=${c.pitch_derived == null ? "—" : fmt(c.pitch_derived)} mm · ${c.modtype_derived}<br>` +
      `t_front=${fmt(c.t_front)} t_back=${fmt(c.t_back)} ns` +
      (c.is_seed ? `<br><b>★ seed</b>` : ""), ev))
    .on("mouseleave", () => tip(null));
  cells.filter((c) => c.is_seed).forEach((c) => {
    svg.append("text").attr("x", x(c.x)).attr("y", y(c.y) + 5).attr("text-anchor", "middle").attr("fill", "var(--seed)").attr("font-size", 16).attr("font-weight", 700).text("★");
  });
  d.clusters.forEach((c) => {
    svg.append("text").attr("x", x(c.x_cluster)).attr("y", y(c.y_cluster) + 6).attr("text-anchor", "middle").attr("fill", "var(--reco)").attr("font-size", 18).text("+");
  });
  d.truth_photons.forEach((p) => {
    svg.append("text").attr("x", x(p.entry_x)).attr("y", y(p.entry_y) + 5).attr("text-anchor", "middle").attr("fill", "var(--truth)").attr("font-size", 15).text("×");
  });
}

async function renderShower() {
  const el = document.getElementById("view-shower");
  loadingInto(el);
  if (S.event == null) { const ov = await api(`/api/files/${S.file}/overview`); S.event = ov.entries[0].event; }
  const d = await api(`/api/files/${S.file}/event/${S.event}`);
  clear(el);
  const cells = [];
  d.clusters.forEach((c) => c.cells.forEach((x) => cells.push(x)));
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h3>${L("tab_shower")} — event ${d.event}</h3>`;
  el.appendChild(card);
  const W = Math.min(700, el.clientWidth - 40), H = 420, m = { l: 64, r: 20, t: 16, b: 48 };
  const svg = d3.select(card).append("svg").attr("width", W).attr("height", H);
  const x = d3.scaleLinear().domain([0, d3.max(cells, (c) => c.front) || 1]).nice().range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, d3.max(cells, (c) => c.back) || 1]).nice().range([H - m.b, m.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y));
  svg.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 12).text("front energy [MeV]");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -H / 2).attr("y", 16).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 12).text("back energy [MeV]");
  const lim = Math.min(x.domain()[1], y.domain()[1]);
  svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(lim)).attr("y2", y(lim)).attr("stroke", "var(--line)").attr("stroke-dasharray", "4 4");
  svg.append("g").selectAll("circle").data(cells).join("circle")
    .attr("cx", (c) => x(c.front)).attr("cy", (c) => y(c.back)).attr("r", (c) => (c.is_seed ? 6 : 3.5))
    .attr("fill", (c) => (c.is_seed ? "var(--seed)" : "var(--accent)")).attr("opacity", 0.75)
    .on("mousemove", (ev, c) => tip(`mod(${c.imodx},${c.jmody})<br>front ${fmt(c.front)} / back ${fmt(c.back)} MeV<br>ratio back/total = ${fmt(c.energy > 0 ? c.back / c.energy : 0)}`, ev))
    .on("mouseleave", () => tip(null));
  const note = document.createElement("div");
  note.className = "explain";
  note.style.marginTop = "12px";
  note.innerHTML = S.lang === "th"
    ? "แต่ละจุดคือหนึ่งเซลล์: แกนนอน = พลังงานส่วนหน้า, แกนตั้ง = ส่วนหลัง. การที่ shower แบ่งพลังงานหน้า/หลังคือการพัฒนาตามความลึก (longitudinal development) — ใช้แยกโฟตอนจากพื้นหลังและช่วย calibrate."
    : "Each point is one cell: x = front-segment energy, y = back-segment energy. The front/back split is the longitudinal shower development — useful to separate photons from background and to calibrate.";
  card.appendChild(note);
}

async function renderDictionary() {
  const el = document.getElementById("view-dictionary");
  loadingInto(el);
  if (!S.dict) S.dict = await api(`/api/dictionary`);
  const schema = await api(`/api/files/${S.file}/schema`);
  const range = {};
  schema.forEach((r) => { range[r.name] = r.sample_min == null ? "" : `${fmt(r.sample_min)} … ${fmt(r.sample_max)}`; });
  clear(el);
  const card = document.createElement("div");
  card.className = "card";
  el.appendChild(card);
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = L("search");
  search.style.marginBottom = "10px";
  card.appendChild(search);
  const tbl = document.createElement("table");
  tbl.className = "dict";
  card.appendChild(tbl);
  const draw = (q) => {
    const rows = S.dict.filter((f) => !q || f.name.toLowerCase().includes(q.toLowerCase()) || t(f.text.what).toLowerCase().includes(q.toLowerCase()));
    tbl.innerHTML =
      `<thead><tr><th>${L("col_name")}</th><th>${L("col_level")}</th><th>${L("col_unit")}</th><th>${L("col_present")}</th><th>${L("col_what")}</th><th>${L("col_note")}</th><th>${L("col_range")}</th></tr></thead>`;
    const tb = document.createElement("tbody");
    rows.forEach((f) => {
      const tr = document.createElement("tr");
      const badge = f.present === "derived" ? `<span class="badge derived">${L("derived")}</span>` : `<span class="badge root">ROOT</span>`;
      tr.innerHTML =
        `<td class="kv">${f.name}${f.formula ? `<br><span class="muted">\\(${f.formula.replace(/_/g, "\\_")}\\)</span>` : ""}</td>` +
        `<td>${f.level}</td><td>${f.unit || "—"}</td><td>${badge}</td>` +
        `<td>${t(f.text.what)}</td><td class="muted">${t(f.note) || ""}</td><td class="kv">${range[f.name] || "—"}</td>`;
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    mathify(tbl);
  };
  search.oninput = () => draw(search.value);
  draw("");
}

function renderTour() {
  const el = document.getElementById("view-tour");
  clear(el);
  const card = document.createElement("div");
  card.className = "card";
  el.appendChild(card);
  const steps = TOUR_STEPS();
  const wrap = document.createElement("div");
  wrap.className = "steps";
  steps.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "step";
    d.innerHTML = `<div class="n">${i + 1}</div><div class="body"><b>${t(s.title)}</b><div class="muted">${t(s.body)}</div></div>`;
    if (s.view) {
      const b = document.createElement("button");
      b.className = "pill";
      b.textContent = s.cta ? t(s.cta) : "→";
      b.onclick = () => setView(s.view);
      d.querySelector(".body").appendChild(b);
    }
    wrap.appendChild(d);
  });
  card.appendChild(wrap);
  const disc = document.createElement("div");
  disc.className = "card";
  disc.innerHTML = `<h3>${S.lang === "th" ? "4 จุดที่ไฟล์จริงต่างจาก instruction" : "4 places the real files differ from the instruction"}</h3>`;
  DISCREPANCIES().forEach((x) => {
    const e = document.createElement("div");
    e.className = "disc";
    e.innerHTML = `<b>${t(x.h)}</b><br>${t(x.b)}`;
    disc.appendChild(e);
  });
  el.appendChild(disc);
}

async function renderExploration() {
  const el = document.getElementById("view-exploration");
  loadingInto(el);
  if (!S.explainers) S.explainers = await api(`/api/explainers`);
  const dist = await api(`/api/files/${S.file}/distributions`);
  clear(el);

  const top = document.createElement("div");
  top.className = "card";
  const why = findEx("why_eda"), thr = findEx("through_line");
  top.innerHTML = `<h3>${L("why_eda_title")}</h3>`;
  top.appendChild(explainPanel(why));
  const thrP = explainPanel(thr);
  thrP.style.marginTop = "8px";
  top.appendChild(thrP);
  el.appendChild(top);
  mathify(top);

  const charts = [
    { id: "truth_spectrum", values: dist.truth_energy_gev, xlabel: L("energy_gev"), logToggle: true, exId: "log_spectrum" },
    { id: "cell_multiplicity", values: dist.n_cells, xlabel: L("n_cells"), bins: 30 },
    { id: "response", values: dist.response.filter((v) => v > -1 && v < 1), xlabel: L("response_x") },
    { id: "dr", values: dist.dr_cluster_truth.filter((v) => v < 200), xlabel: L("dr_x") },
  ];
  charts.forEach((c) => {
    const row = document.createElement("div");
    row.className = "row";
    const left = document.createElement("div");
    left.className = "card grow";
    const ex = findEx(c.id);
    left.innerHTML = `<h3>${t(ex.title)}</h3>`;
    const right = document.createElement("div");
    right.className = "grow";
    right.style.maxWidth = "420px";
    right.appendChild(explainPanel(ex));
    if (c.exId) { const x2 = explainPanel(findEx(c.exId)); x2.style.marginTop = "8px"; right.appendChild(x2); }
    row.appendChild(left);
    row.appendChild(right);
    el.appendChild(row);
    mathify(right);
    histogram(left, c.values, { xlabel: c.xlabel, bins: c.bins || 40, logToggle: c.logToggle });
  });
}

function histogram(card, values, opts) {
  const holder = document.createElement("div");
  card.appendChild(holder);
  let logx = false;
  const draw = () => {
    clear(holder);
    if (opts.logToggle) {
      const bar = document.createElement("div");
      bar.className = "toolbar";
      ["lin", "log"].forEach((mode) => {
        const b = document.createElement("button");
        b.className = "pill" + ((mode === "log") === logx ? " active" : "");
        b.textContent = L(mode);
        b.onclick = () => { logx = mode === "log"; draw(); };
        bar.appendChild(b);
      });
      holder.appendChild(bar);
    }
    const W = Math.max(280, Math.min(560, card.clientWidth - 28)), H = 300, m = { l: 52, r: 16, t: 12, b: 42 };
    const vals = values.filter((v) => isFinite(v) && (!logx || v > 0));
    let x;
    if (logx) {
      const e = d3.extent(vals);
      x = d3.scaleLog().domain([Math.max(e[0], 1e-3), e[1]]).range([m.l, W - m.r]);
    } else {
      x = d3.scaleLinear().domain(d3.extent(vals)).nice().range([m.l, W - m.r]);
    }
    const thresholds = logx
      ? d3.range(opts.bins + 1).map((i) => x.invert(m.l + (i / opts.bins) * (W - m.r - m.l)))
      : x.ticks(opts.bins);
    const bins = d3.bin().domain(x.domain()).thresholds(thresholds)(vals);
    const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) || 1]).nice().range([H - m.b, m.t]);
    const svg = d3.select(holder).append("svg").attr("width", W).attr("height", H);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6, logx ? "~s" : undefined));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append("g").selectAll("rect").data(bins).join("rect")
      .attr("x", (b) => x(b.x0) + 1).attr("y", (b) => y(b.length))
      .attr("width", (b) => Math.max(1, x(b.x1) - x(b.x0) - 1))
      .attr("height", (b) => y(0) - y(b.length)).attr("fill", "var(--accent)").attr("opacity", 0.8);
    svg.append("text").attr("x", W / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 11).text(opts.xlabel);
    svg.append("text").attr("transform", "rotate(-90)").attr("x", -H / 2).attr("y", 14).attr("text-anchor", "middle").attr("fill", "var(--muted)").attr("font-size", 11).text(L("counts"));
  };
  draw();
}

function TOUR_STEPS() {
  const S2 = (th, en) => ({ th, en });
  return [
    { title: S2("1. โฟตอนถือกำเนิด", "1. A photon is born"), body: S2("ในการชนกัน อนุภาคถูกสร้างที่ production vertex (sig_flux_prod_vertex_*) พร้อมโมเมนตัม (px,py,pz)", "In the collision a particle is produced at the production vertex (sig_flux_prod_vertex_*) with momentum (px,py,pz)") },
    { title: S2("2. บินไปตามลำบีม", "2. It flies along the beam"), body: S2("ทิศทางคือ dx/dz=px/pz, dy/dz=py/pz จนถึงหน้า ECAL ที่ z≈12.62 m", "Direction dx/dz=px/pz, dy/dz=py/pz until it reaches the ECAL face at z≈12.62 m") },
    { title: S2("3. เข้าหน้า ECAL", "3. It enters the ECAL face"), body: S2("จุดเข้าจริงคือ (sig_flux_entry_x, sig_flux_entry_y) — ดูตำแหน่งจริงบนเซนเซอร์", "The true entry is (sig_flux_entry_x, sig_flux_entry_y) — see the real spot on the sensor"), view: "sensor", cta: S2("เปิดหน้าเซนเซอร์", "open detector") },
    { title: S2("4. เกิด shower", "4. It showers"), body: S2("โฟตอนหนึ่งตัวกระจายพลังงานไปหลายเซลล์ (electromagnetic shower) พีคที่ seed แล้วจางลง", "One photon spreads energy over many cells (electromagnetic shower), peaking at the seed and fading out") },
    { title: S2("5. หน้า/หลัง", "5. front / back"), body: S2("ทุกเซลล์อ่านพลังงานสองชั้น: front + back = energy — การพัฒนาตามความลึก", "Every cell reads two layers: front + back = energy — the longitudinal development"), view: "shower", cta: S2("ดูความลึก", "see depth") },
    { title: S2("6. รวมเป็น cluster", "6. grouped into a cluster"), body: S2("reconstruction รวมเซลล์เป็น cluster หนึ่งก้อน (x_cluster, y_cluster, total_energy)", "reconstruction groups the cells into one cluster (x_cluster, y_cluster, total_energy)") },
    { title: S2("7. จับคู่กับความจริง", "7. truth-matched"), body: S2("เก็บ cluster ที่ seed ใกล้จุดเข้าจริงที่สุด → กลายเป็นหนึ่ง entry (1 โฟตอน + 1 cluster)", "keep the cluster whose seed is closest to the true entry → becomes one entry (1 photon + 1 cluster)") },
    { title: S2("8. สำรวจ & สูตร", "8. explore & formulas"), body: S2("ทำไมแต่ละกราฟถึงสำคัญ และสูตรเบื้องหลัง", "why each plot matters and the formulas behind them"), view: "exploration", cta: S2("ไปหน้า สำรวจ", "go to exploration") },
  ];
}

function DISCREPANCIES() {
  const S2 = (th, en) => ({ th, en });
  return [
    { h: S2("event ↔ entry", "event ↔ entry"), b: S2("instruction บอก 1 event = 1 cluster แต่จริงมีหลาย entry ต่อ event (cluster เดียวจับคู่หลายโฟตอน) — entry คือแถวจริง", "instruction says 1 event = 1 cluster, but real files store several entries per event (one cluster matched to several photons) — the entry is the real row") },
    { h: S2("เก็บแบบ jagged ไม่ใช่ padded", "jagged, not padded"), b: S2("มี n* counter แต่ array เป็น variable-length จริง ๆ (ak.num = จำนวนจริง ไม่ต้องตัด zero-padding)", "n* counters exist but arrays are truly variable-length (ak.num = real count, no zero-padding to strip)") },
    { h: S2("หน่วยเป็น GeV ไม่ใช่ MeV", "units are GeV, not MeV"), b: S2("instruction เขียน MeV แต่ sig_flux_eTot เป็น GeV (สูงสุด 203 เทียบ pz สูงสุด ~203000 MeV); cell energy เป็น MeV", "instruction says MeV but sig_flux_eTot is GeV (max 203 vs pz max ~203000 MeV); cell energies are MeV") },
    { h: S2("ฟิลด์ที่ต้องคำนวณเอง", "fields we must derive"), b: S2("cell_pitch, cell_modType, cell_rel_* ไม่มีในไฟล์ — คำนวณจาก geometry และติดป้าย 'derived'", "cell_pitch, cell_modType, cell_rel_* are absent — derived from geometry and labelled 'derived'") },
  ];
}

async function init() {
  document.getElementById("lang-th").onclick = () => switchLang("th");
  document.getElementById("lang-en").onclick = () => switchLang("en");
  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  applyStaticI18n();
  const files = await api("/api/files");
  const sel = document.getElementById("file-select");
  files.forEach((f) => { const o = document.createElement("option"); o.value = f.name; o.textContent = `${f.name} (${f.dataset}, ${f.n_entries})`; sel.appendChild(o); });
  sel.onchange = () => { S.file = sel.value; S.event = null; render(); };
  S.file = files[0].name;
  render();
}

function switchLang(lang) {
  S.lang = lang;
  document.getElementById("lang-th").classList.toggle("active", lang === "th");
  document.getElementById("lang-en").classList.toggle("active", lang === "en");
  applyStaticI18n();
  render();
}

window.addEventListener("DOMContentLoaded", init);
