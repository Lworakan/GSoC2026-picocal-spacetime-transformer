window.ThreeView = (function () {
  const THREE = window.THREE;
  let cur = null;

  const FACE_Z = 0, FRONT_DEPTH = 150, GAP = 18, BACK_DEPTH = 150;
  const BACK_Z = FACE_Z + FRONT_DEPTH + GAP;
  const VERTEX_Z = -1500;
  const C_MM_NS = 299.792458;

  function gammaln(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp); let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) { y++; ser += c[j] / y; }
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function lowerP(a, x) {
    if (x <= 0) return 0;
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 300; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-11) break; }
      return sum * Math.exp(-x + a * Math.log(x) - gammaln(a));
    }
    const FPMIN = 1e-300; let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
    for (let i = 1; i < 300; i++) {
      const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-11) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - gammaln(a)) * h;
  }
  const Ec = 0.012, bRate = 0.5, X0 = 9;
  function longitudinal(E) {
    const tmax = Math.log(Math.max(E, Ec * 1.1) / Ec) + 0.5;
    const a = Math.max(1.05, bRate * tmax + 1);
    const cF = lowerP(a, bRate * (FRONT_DEPTH / X0));
    const cT = lowerP(a, bRate * ((FRONT_DEPTH + BACK_DEPTH) / X0));
    return { fFront: cF, fBack: cT - cF, fLeak: 1 - cT };
  }
  function lateral(r) { return 0.7 * Math.exp(-r / 11) / 121 + 0.3 * Math.exp(-r / 42) / 1764; }

  function heatE(t) {
    t = Math.min(1, Math.max(0, t));
    const s = [[0.10, 0.02, 0.28], [0.13, 0.32, 0.56], [0.10, 0.66, 0.55], [0.62, 0.85, 0.20], [0.98, 0.92, 0.30]];
    const f = t * (s.length - 1), i = Math.floor(f), k = f - i, a = s[i], b = s[Math.min(i + 1, s.length - 1)];
    return new THREE.Color(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k);
  }
  function coolwarm(t) {
    t = Math.min(1, Math.max(0, t));
    return new THREE.Color(0.23 + 0.75 * t, 0.30 + 0.35 * Math.sin(Math.PI * t), 0.95 - 0.75 * t);
  }

  function L(lang, th, en) { return lang === "th" ? th : en; }

  function unmount() {
    if (!cur) return;
    cancelAnimationFrame(cur.raf);
    window.removeEventListener("resize", cur.onResize);
    if (cur.renderer) { cur.renderer.forceContextLoss && cur.renderer.forceContextLoss(); cur.renderer.dispose(); if (cur.renderer.domElement && cur.renderer.domElement.parentNode) cur.renderer.domElement.parentNode.removeChild(cur.renderer.domElement); }
    cur = null;
  }

  function mount(container, data, lang) {
    unmount();
    container.innerHTML = "";
    container.style.position = "relative";
    const W0 = container.clientWidth || 900, H0 = Math.max(420, Math.round(Math.min(640, window.innerHeight * 0.66)));
    container.style.height = H0 + "px";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    scene.fog = new THREE.Fog(0x070b14, 1400, 3600);
    const camera = new THREE.PerspectiveCamera(42, W0 / H0, 1, 9000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const key = new THREE.DirectionalLight(0xdfe9ff, 0.5); key.position.set(-400, 600, 800); scene.add(key);

    const cl = (data.clusters && data.clusters[0]) || { cells: [], x_cluster: 0, y_cluster: 0, total_energy_front: 0, total_energy_back: 0, total_energy: 0, n_cells: 0 };
    const cells = cl.cells || [];
    const seedCell = cells.find((c) => c.is_seed) || cells[0] || { x: cl.x_cluster, y: cl.y_cluster };
    const ox = seedCell.x, oy = seedCell.y;
    const truths = data.truth_photons || [];
    const primary = truths[0] || { entry_x: ox, entry_y: oy, entry_z: 12620, energy_gev: 1, dxdz: 0, dydz: 0, prod_z: 0, timing: 0 };

    const HALF = Math.max(120, Math.max(...cells.map((c) => Math.max(Math.abs(c.x - ox), Math.abs(c.y - oy))), 60) + 80);

    function slab(z0, depth, col) {
      const g = new THREE.BoxGeometry(HALF * 2, HALF * 2, depth);
      const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.04 }));
      mesh.position.set(0, 0, z0 + depth / 2); scene.add(mesh);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.28 }));
      e.position.copy(mesh.position); scene.add(e);
    }
    slab(FACE_Z, FRONT_DEPTH, 0x3f6fbf);
    slab(BACK_Z, BACK_DEPTH, 0x7a5bd0);

    const deposits = new THREE.Group(); scene.add(deposits);
    const overlay = new THREE.Group(); scene.add(overlay);
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);

    const state = { color: "energy", idealize: false, E: primary.energy_gev, spin: true };

    function idealCell(c, layer) {
      const ax = primary.entry_x + (layer === "back" ? primary.dxdz * (BACK_Z) : 0);
      const ay = primary.entry_y + (layer === "back" ? primary.dydz * (BACK_Z) : 0);
      const r = Math.hypot(c.x - primary.entry_x, c.y - primary.entry_y);
      const lon = longitudinal(state.E);
      const w = lateral(r) * (c.pitch_derived || 30) * (c.pitch_derived || 30) * (layer === "back" ? 1.15 : 1);
      return state.E * 1000 * (layer === "front" ? lon.fFront : lon.fBack) * w;
    }

    function build() {
      for (let i = deposits.children.length - 1; i >= 0; i--) { const c = deposits.children[i]; deposits.remove(c); if (c.material) c.material.dispose(); }
      for (let i = overlay.children.length - 1; i >= 0; i--) { const c = overlay.children[i]; overlay.remove(c); if (c.material) c.material.dispose(); if (c.geometry && c.geometry !== boxGeo) c.geometry.dispose(); }

      const fronts = [], backs = [];
      cells.forEach((c) => {
        const f = state.idealize ? idealCell(c, "front") : c.front;
        const b = state.idealize ? idealCell(c, "back") : c.back;
        fronts.push(f); backs.push(b);
      });
      const emax = Math.max(1e-6, ...fronts, ...backs);
      const logMax = Math.log(emax), logFloor = Math.log(Math.max(1e-3, emax / 1e4));

      function drawLayer(vals, z0, depth, layer) {
        cells.forEach((c, i) => {
          const e = vals[i]; if (!(e > Math.exp(logFloor))) return;
          const lt = Math.min(1, Math.max(0, (Math.log(e) - logFloor) / (logMax - logFloor)));
          const h = depth * (0.12 + 0.88 * lt);
          const p = c.pitch_derived || 30;
          const m = new THREE.Mesh(boxGeo);
          m.scale.set(p * 0.82, p * 0.82, h);
          m.position.set(c.x - ox, c.y - oy, z0 + h / 2);
          let col;
          if (state.color === "energy") col = heatE(lt);
          else { const tt = layer === "back" ? c.t_back : c.t_front; col = coolwarm((tt % 5) / 5); }
          m.material = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.55 + 0.45 * lt });
          deposits.add(m);
        });
      }
      drawLayer(fronts, FACE_Z, FRONT_DEPTH, "front");
      drawLayer(backs, BACK_Z, BACK_DEPTH, "back");

      truths.forEach((p, idx) => {
        const entry = new THREE.Vector3(p.entry_x - ox, p.entry_y - oy, FACE_Z);
        const vtx = new THREE.Vector3(entry.x - p.dxdz * (FACE_Z - VERTEX_Z), entry.y - p.dydz * (FACE_Z - VERTEX_Z), VERTEX_Z);
        const lg = new THREE.BufferGeometry().setFromPoints([vtx, entry]);
        const line = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: idx === 0 ? 0x67e8f9 : 0xffb36b, dashSize: 34, gapSize: 20, transparent: true, opacity: 0.85 }));
        line.computeLineDistances(); overlay.add(line);
        const v = new THREE.Mesh(new THREE.SphereGeometry(idx === 0 ? 13 : 10, 16, 16), new THREE.MeshBasicMaterial({ color: idx === 0 ? 0xffd24a : 0xffb36b }));
        v.position.copy(vtx); overlay.add(v);
        addCross(entry, 26, idx === 0 ? 0xff5d6c : 0xff9d6c);
      });
      addPlus(new THREE.Vector3(cl.x_cluster - ox, cl.y_cluster - oy, FACE_Z + 2), 30, 0xffffff);
      if (seedCell) addStar(new THREE.Vector3(seedCell.x - ox, seedCell.y - oy, FACE_Z + 3));

      paint();
    }

    function addCross(pos, s, col) {
      const m = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.8 });
      const p = [pos.x - s, pos.y - s, pos.z + 1, pos.x + s, pos.y + s, pos.z + 1, pos.x - s, pos.y + s, pos.z + 1, pos.x + s, pos.y - s, pos.z + 1];
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
      overlay.add(new THREE.LineSegments(g, m));
    }
    function addPlus(pos, s, col) {
      const m = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.95 });
      const p = [pos.x - s, pos.y, pos.z, pos.x + s, pos.y, pos.z, pos.x, pos.y - s, pos.z, pos.x, pos.y + s, pos.z];
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
      overlay.add(new THREE.LineSegments(g, m));
    }
    function addStar(pos) {
      const m = new THREE.PointsMaterial({ color: 0xffd24a, size: 16, sizeAttenuation: false });
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute([pos.x, pos.y, pos.z], 3));
      overlay.add(new THREE.Points(g, m));
    }

    function paint() {
      const tof = Math.hypot(primary.entry_x, primary.entry_y, primary.entry_z) / C_MM_NS;
      const tf = cl.total_energy_front, tb = cl.total_energy_back, tt = tf + tb || 1;
      const captured = primary.energy_gev > 0 ? cl.total_energy / (primary.energy_gev * 1000) : 0;
      const dr = Math.hypot(cl.x_cluster - primary.entry_x, cl.y_cluster - primary.entry_y);
      const pf = Math.round((tf / tt) * 100), pb = 100 - pf;
      con.innerHTML =
        `<h2>${L(lang, "ค่าที่อ่านได้จาก event", "event readout")}</h2>` +
        row(L(lang, "เวลาเดินทาง d/c", "time of flight d/c"), tof.toFixed(1) + " ns") +
        row(L(lang, "พลังงานจริง (truth)", "truth energy"), primary.energy_gev.toFixed(2) + " GeV") +
        row(L(lang, "เซลล์ที่ติด (tokens)", "lit cells (tokens)"), cl.n_cells + (truths.length > 1 ? " (merged)" : "")) +
        row(L(lang, "reco − truth", "reco − truth"), dr.toFixed(1) + " mm") +
        row(L(lang, "reco/truth", "reco/truth"), (captured * 100).toFixed(0) + "%") +
        `<div class="longbar"><i class="lf" style="width:${pf}%"></i><i class="lb" style="width:${pb}%"></i></div>` +
        `<div class="longkey"><i><span class="dot lf"></span>front ${pf}%</i><i><span class="dot lb"></span>back ${pb}%</i></div>`;
    }
    function row(k, v) { return `<div class="stat"><span>${k}</span><b>${v}</b></div>`; }

    const ctrl = document.createElement("div"); ctrl.className = "panel tv-controls";
    ctrl.innerHTML =
      `<div class="tv-row"><div class="seg" id="tvColor"><button data-v="energy" class="on">${L(lang, "พลังงาน", "Energy")}</button><button data-v="time">${L(lang, "เวลา", "Time")}</button></div></div>` +
      `<div class="tv-row"><label class="tvtog" id="tvIdeal"><span>${L(lang, "โหมด idealize (shower ในอุดมคติ)", "idealize (textbook shower)")}</span><span class="sw"></span></label></div>` +
      `<div class="tv-row" id="tvErow" style="display:none"><label class="lbl">${L(lang, "พลังงานสมมติ", "model energy")} <b id="tvE"></b></label><input id="tvEslider" type="range" min="0" max="1000" value="500"></div>` +
      `<div class="tv-row"><label class="tvtog on" id="tvSpin"><span>${L(lang, "หมุนอัตโนมัติ", "auto-rotate")}</span><span class="sw"></span></label></div>`;
    container.appendChild(ctrl);

    const con = document.createElement("div"); con.className = "panel tv-console mono"; container.appendChild(con);

    const leg = document.createElement("div"); leg.className = "panel tv-legend";
    leg.innerHTML =
      `<h2>${L(lang, "นี่คือ branch อะไรบ้าง", "branch ↔ glyph")}</h2>` +
      legRow("#ffd24a", "50%", L(lang, "จุดกำเนิดโฟตอน", "photon birth"), "sig_flux_prod_vertex_*") +
      legRow("linear-gradient(90deg,#67e8f9,#2a5)", "3px", L(lang, "ทิศ + เวลา", "flight + timing"), "sig_flux_px/py/pz · timing") +
      legGlyph("✕", "#ff5d6c", L(lang, "จุดเข้า ECAL", "ECAL entry"), "sig_flux_entry_x/y/z") +
      legRow("#48b0ff", "13px", L(lang, "พลังงานชั้นหน้า", "front-layer energy"), "cell_energies_front · cell_x/y") +
      legRow("#9d6bff", "13px", L(lang, "พลังงานชั้นหลัง", "back-layer energy"), "cell_energies_back") +
      legGlyph("+", "#fff", L(lang, "ตำแหน่ง cluster (reco)", "reco cluster centre"), "x_cluster / y_cluster") +
      legGlyph("★", "#ffd24a", L(lang, "เซลล์ seed", "seed cell"), "max-energy cell");
    container.appendChild(leg);

    function legRow(bg, h, ds, br) {
      return `<div class="leg"><span class="glyph mk" style="background:${bg};height:${h};${h === "3px" ? "margin-top:6px;width:13px" : ""}"></span><div><span class="ds">${ds}</span><br><span class="br mono">${br}</span></div></div>`;
    }
    function legGlyph(ch, col, ds, br) {
      return `<div class="leg"><span class="mk" style="color:${col};font-weight:800;font-size:15px;line-height:13px">${ch}</span><div><span class="ds">${ds}</span><br><span class="br mono">${br}</span></div></div>`;
    }

    ctrl.querySelectorAll("#tvColor button").forEach((b) => b.addEventListener("click", () => {
      ctrl.querySelectorAll("#tvColor button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); state.color = b.dataset.v; build();
    }));
    const eRow = ctrl.querySelector("#tvErow"), eSlider = ctrl.querySelector("#tvEslider"), eLab = ctrl.querySelector("#tvE");
    function sliderToE(v) { const t = v / 1000; return Math.exp(Math.log(1) * (1 - t) + Math.log(200) * t); }
    eLab.textContent = state.E.toFixed(1) + " GeV";
    ctrl.querySelector("#tvIdeal").addEventListener("click", function () {
      this.classList.toggle("on"); state.idealize = this.classList.contains("on");
      eRow.style.display = state.idealize ? "block" : "none"; build();
    });
    eSlider.addEventListener("input", function () { state.E = sliderToE(+this.value); eLab.textContent = state.E.toFixed(1) + " GeV"; build(); });
    ctrl.querySelector("#tvSpin").addEventListener("click", function () { this.classList.toggle("on"); state.spin = this.classList.contains("on"); });

    const target = new THREE.Vector3(0, 0, (FRONT_DEPTH + BACK_Z + BACK_DEPTH) / 2);
    let az = -0.9, pol = 1.1, dist = Math.max(900, HALF * 3.4), dragging = false, px = 0, py = 0;
    function applyCam() {
      const sp = Math.sin(pol), cp = Math.cos(pol), sa = Math.sin(az), ca = Math.cos(az);
      camera.position.set(target.x + dist * sp * sa, target.y + dist * cp, target.z + dist * sp * ca);
      camera.lookAt(target);
    }
    const el = renderer.domElement;
    el.addEventListener("pointerdown", (e) => { dragging = true; px = e.clientX; py = e.clientY; state.spin = false; const t = ctrl.querySelector("#tvSpin"); t.classList.remove("on"); el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => { if (!dragging) return; az -= (e.clientX - px) * 0.006; pol -= (e.clientY - py) * 0.006; pol = Math.max(0.18, Math.min(Math.PI - 0.18, pol)); px = e.clientX; py = e.clientY; applyCam(); });
    el.addEventListener("pointerup", () => { dragging = false; });
    el.addEventListener("wheel", (e) => { e.preventDefault(); dist *= Math.pow(1.0015, e.deltaY); dist = Math.max(500, Math.min(4000, dist)); applyCam(); }, { passive: false });

    function onResize() {
      const w = container.clientWidth || W0;
      camera.aspect = w / H0; camera.updateProjectionMatrix(); renderer.setSize(w, H0);
    }
    window.addEventListener("resize", onResize);

    build(); applyCam();
    const self = { renderer, raf: 0, onResize };
    function loop() { self.raf = requestAnimationFrame(loop); if (state.spin && !dragging) { az += 0.0016; applyCam(); } renderer.render(scene, camera); }
    cur = self; loop();
  }

  return { mount, unmount };
})();
