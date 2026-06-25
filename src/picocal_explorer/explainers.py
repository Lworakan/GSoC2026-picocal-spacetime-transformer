from __future__ import annotations


def _e(eid, title_th, title_en, formula_latex, what_th, what_en, why_th, why_en, ml_th, ml_en):
    return {
        "id": eid,
        "title": {"th": title_th, "en": title_en},
        "formula_latex": formula_latex,
        "what": {"th": what_th, "en": what_en},
        "why": {"th": why_th, "en": why_en},
        "ml_analogy": {"th": ml_th, "en": ml_en},
    }


PLOTS = [
    _e(
        "cell_multiplicity", "จำนวนเซลล์ต่อ cluster", "Cells per cluster", "n = \\text{len}(\\text{energy})",
        "นับจำนวนเซลล์ในแต่ละ cluster", "Count of cells in each cluster",
        "แต่ละเซลล์ = 1 token ของ Transformer → นี่คือการกระจายของ sequence length บอกว่าต้อง pad/mask แค่ไหน",
        "Each cell is one Transformer token, so this is the sequence-length distribution that sets padding/masking",
        "เหมือนดูการกระจายความยาวประโยคก่อนตั้ง max_seq_len ใน NLP",
        "Like checking sentence-length distribution before setting max_seq_len in NLP",
    ),
    _e(
        "truth_spectrum", "สเปกตรัมพลังงานจริง", "Truth-energy spectrum", "dN/dE",
        "ฮิสโทแกรมของ target (sig_flux_eTot)", "Histogram of the target (sig_flux_eTot)",
        "รู้รูปร่าง/ช่วงของสิ่งที่จะทำนาย → ช่วยตัดสินใจว่าควรทำนาย log E ไหม",
        "Reveals the shape/range of what we predict, guiding the choice to regress log E",
        "เหมือนดู label distribution ในงาน regression",
        "Like inspecting the label distribution in a regression task",
    ),
    _e(
        "response", "Energy response ΔE/E", "Energy response ΔE/E",
        "\\frac{\\Delta E}{E}=\\frac{E_{\\text{reco}}-E_{\\text{true}}}{E_{\\text{true}}}",
        "error เชิงสัดส่วนของพลังงานที่ reco เทียบกับความจริง",
        "relative error of reconstructed vs true energy",
        "ค่ากลาง = bias, ความกว้าง = resolution; เป็น figure of merit หลักของการ reconstruct",
        "Its mean is the bias and its width is the resolution — the main figure of merit",
        "เหมือน relative error / MAPE แทน absolute error",
        "Like relative error / MAPE instead of absolute error",
    ),
    _e(
        "dr", "ระยะ cluster ↔ truth", "Cluster ↔ truth distance",
        "dr=\\sqrt{\\Delta x^2+\\Delta y^2}",
        "ระยะ Euclidean บนหน้า ECAL ระหว่าง cluster ที่ reco กับจุดเข้าจริง",
        "Euclidean distance on the ECAL face between reco cluster and true entry",
        "ตรวจคุณภาพการจับคู่: dr เล็ก = match ดี, หางยาว = label noise",
        "Checks matching quality: small dr = good match, a long tail = label noise",
        "เหมือนตรวจ label noise ก่อนเทรน",
        "Like auditing label noise before training",
    ),
    _e(
        "seed", "Seed = เซลล์พลังงานสูงสุด", "Seed = max-energy cell", "i_{\\text{seed}}=\\arg\\max_i E_i",
        "เลือกเซลล์พลังงานสูงสุดเป็นจุดเริ่มของ cluster",
        "pick the highest-energy cell as the cluster starting point",
        "เป็น baseline ระบุตำแหน่งโฟตอน และขั้นแรกของ clustering",
        "A baseline for locating the photon and the first step of clustering",
        "เหมือน baseline model ง่าย ๆ ก่อนโมเดลจริง",
        "Like a simple baseline model before the real one",
    ),
    _e(
        "efficiency", "Efficiency เทียบรัศมี", "Efficiency vs radius", "\\epsilon(R)=\\frac{N_{\\text{found}}(R)}{N_{\\text{total}}}",
        "สัดส่วน cluster ที่พบเซลล์พลังงานสูงสุดภายในรัศมี R ของจุดจริง",
        "fraction of clusters with a max-energy cell within radius R of the true entry",
        "ใช้เลือกขนาดหน้าต่าง/รัศมีที่เหมาะสม",
        "Helps choose a sensible window radius",
        "เหมือน accuracy/recall เป็นฟังก์ชันของ threshold",
        "Like accuracy/recall as a function of a threshold",
    ),
]

FORMULAS = [
    _e(
        "resolution", "Energy resolution", "Energy resolution",
        "\\frac{\\sigma_E}{E}=\\frac{a}{\\sqrt{E}}\\oplus b\\oplus\\frac{c}{E}",
        "ความกว้างของ ΔE/E เป็นฟังก์ชันของพลังงาน (⊕ = บวกแบบ quadrature)",
        "the width of ΔE/E as a function of energy (⊕ = sum in quadrature)",
        "a = stochastic (ผันผวนของ shower, เด่นที่ E ต่ำ), b = constant (คาลิเบรชัน, เด่นที่ E สูง), c = noise → จึงต้องดู performance แยกตามช่วงพลังงาน",
        "a = stochastic (shower fluctuations, dominant at low E), b = constant (calibration, dominant at high E), c = noise — so evaluate performance binned by energy",
        "เหมือน error ที่ไม่คงที่ตาม scale ของ input",
        "Like an error that varies with input scale",
    ),
    _e(
        "log_spectrum", "Log energy spectrum", "Log energy spectrum",
        "\\log\\frac{dN}{dE}=-n\\log E+c",
        "สเปกตรัมพลังงานมักตกแบบ power law; ใส่ log แล้วเป็นเส้นตรง",
        "energy spectra often fall as a power law; on a log scale they become a straight line",
        "เห็นช่วงไดนามิกกว้าง (≈0.1→200 GeV) และนำไปสู่การทำนาย log E แทน E",
        "Reveals the wide dynamic range (≈0.1→200 GeV) and motivates predicting log E instead of E",
        "เหมือน log-transform target ที่มีหางหนัก",
        "Like log-transforming a heavy-tailed target",
    ),
    _e(
        "why_eda", "ทำไมต้อง EDA ทั้งที่มี guideline", "Why EDA even with a guideline", "",
        "guideline บอกว่าข้อมูลควรเป็นแบบไหน; EDA พิสูจน์ว่าจริง ๆ เป็นแบบนั้นไหม",
        "the guideline says what the data should be; EDA proves what it actually is",
        "การสำรวจจริงเจอ 4 จุดที่ไฟล์ต่างจาก instruction (หน่วย, jagged, event/entry, ฟิลด์ที่หาย) ซึ่งถ้าเชื่อ guideline ดื้อ ๆ โมเดลจะพัง",
        "exploration surfaced four mismatches with the instruction (units, jagged arrays, event/entry, missing fields) that would silently break a model",
        "เหมือน EDA ก่อนเทรนเพื่อกัน data bug",
        "Like EDA before training to catch data bugs",
    ),
    _e(
        "why_no_inverse", "ทำไม invert การ simulate ไม่ได้", "Why we cannot invert the simulation", "",
        "การ simulate เป็นกระบวนการสุ่ม (shower fluctuations) จากพลังงานจริง → รูปแบบเซลล์",
        "simulation is a stochastic forward map from true energy to a pattern of cell energies",
        "มันเป็น many-to-one + มี randomness จึงไม่มี inverse แบบปิด เราจึงเรียนรู้ตัวประมาณเชิงสถิติแทน",
        "it is many-to-one and random, so there is no closed-form inverse — we learn a statistical estimator instead",
        "เหมือนเหตุผลที่เราเทรนโมเดลแทนการแก้สมการย้อนกลับ",
        "Like why we train a model instead of solving the inverse equation",
    ),
    _e(
        "why_transformer", "ทำไมต้อง Transformer", "Why a Transformer", "",
        "เซลล์ในแต่ละ cluster เป็นเซตความยาวไม่คงที่ (granularity แปรผัน) แต่ละเซลล์มีตำแหน่งและเวลา",
        "the cells in a cluster form a variable-length set (granularity varies); each has a position and a time",
        "Transformer รับเซตความยาวแปรผัน เป็น permutation-invariant และใช้ตำแหน่ง+เวลาเป็น spacetime token ผ่าน attention — ตรงข้ามกับ CNN ที่ต้องกริดคงที่",
        "a Transformer handles variable-length sets, is permutation-invariant, and uses position+time as spacetime tokens via attention — unlike a CNN that needs a fixed grid",
        "เหมือนมอง cell เป็น token เหมือนคำในประโยค",
        "Like treating each cell as a token, as words in a sentence",
    ),
    _e(
        "through_line", "เส้นเรื่องที่ร้อยทุกอย่าง", "The through-line", "\\log E \\;\\Leftrightarrow\\; \\Delta E/E \\;\\Leftrightarrow\\; \\sigma_E/E",
        "ทำนาย log E ⟺ สนใจ error เชิงสัดส่วน ⟺ ตรงกับ resolution ที่เป็นปริมาณสัดส่วน",
        "predicting log E ⟺ caring about relative error ⟺ matching the relative resolution",
        "สามสิ่งนี้คือเรื่องเดียวกันมองคนละมุม — คือ 'ทำไม' เบื้องหลังกราฟสำรวจทั้งหมด",
        "these three are one idea seen from different angles — the 'why' behind all the exploration plots",
        "เหมือนเลือก loss/representation ให้เข้ากับธรรมชาติของ target",
        "Like matching loss/representation to the nature of the target",
    ),
]


def _fund(fid, title_th, title_en, body_th, body_en):
    return {"id": fid, "title": {"th": title_th, "en": title_en}, "body": {"th": body_th, "en": body_en}}


FUNDAMENTALS = [
    _fund(
        "lhcb_tour", "LHCb detector ทั้งตัว (โฟตอนผ่านอะไรบ้าง)", "The whole LHCb detector",
        "LHCb เป็น <b>forward spectrometer</b> ที่ LHC — เน้นวัดอนุภาคที่พุ่งไปข้างหน้าตามลำบีม. "
        "เรียงจากจุดชนออกไป: <b>VELO</b> (หาจุดชน/จุดสลาย) → <b>tracker</b> (วัดรอยอนุภาคมีประจุ) → "
        "<b>dipole magnet</b> (ดัดอนุภาคมีประจุ เพื่อวัดโมเมนตัมจากความโค้ง) → <b>RICH</b> (แยกชนิดอนุภาค, PID) → "
        "<b>calorimeters</b>: ECAL (=PicoCal วัดพลังงาน e/γ) แล้วตามด้วย HCAL → <b>muon system</b>. "
        "โฟตอนเป็นกลาง จึงผ่าน tracker/magnet โดยไม่ทิ้งรอย แล้วมาปล่อยพลังงานที่ ECAL — จุดที่เราวัดมัน.",
        "LHCb is a <b>forward spectrometer</b> at the LHC — it instruments particles going forward along the "
        "beam. From the collision outward: <b>VELO</b> (finds collision/decay vertices) → <b>tracker</b> "
        "(measures charged-particle tracks) → <b>dipole magnet</b> (bends charged particles so curvature gives "
        "momentum) → <b>RICH</b> (particle ID) → <b>calorimeters</b>: ECAL (=PicoCal, measures e/γ energy) then "
        "HCAL → <b>muon system</b>. A photon is neutral, so it crosses the tracker/magnet without leaving a "
        "track and deposits its energy in the ECAL — where we measure it.",
    ),
    _fund(
        "coordinates", "ระบบพิกัด & pseudorapidity η", "Coordinates & pseudorapidity η",
        "แกน <b>z</b> = ตามลำบีม (อนุภาคพุ่งไป +z); <b>x, y</b> = ระนาบขวาง (transverse) ที่หน้า ECAL — "
        "คือ cell_x, cell_y. โมเมนตัมตามขวาง \\(p_T=\\sqrt{p_x^2+p_y^2}\\). "
        "<b>Pseudorapidity</b> \\(\\eta=-\\ln\\tan(\\theta/2)\\) (θ = มุมจากลำบีม): η สูง = เฉียดลำบีม = forward. "
        "LHCb ครอบคลุมช่วง forward (η ≈ 2–5) ที่ผลิตอนุภาค heavy-flavour เยอะ. "
        "ค่า dx/dz=px/pz, dy/dz=py/pz บอกความเอียงของรอยที่เข้าหน้า ECAL.",
        "<b>z</b> = along the beam (particles go +z); <b>x, y</b> = the transverse plane at the ECAL face — i.e. "
        "cell_x, cell_y. Transverse momentum \\(p_T=\\sqrt{p_x^2+p_y^2}\\). "
        "<b>Pseudorapidity</b> \\(\\eta=-\\ln\\tan(\\theta/2)\\) (θ = angle from the beam): large η = close to the "
        "beam = forward. LHCb covers the forward region (η ≈ 2–5), where many heavy-flavour particles are "
        "produced. dx/dz=px/pz and dy/dz=py/pz give the track's slope into the ECAL face.",
    ),
    _fund(
        "where_from", "1. โฟตอนมาจากไหน", "1. Where the photon comes from",
        "ที่ LHCb โปรตอนสองลำชนกัน (pp collision) ที่พลังงานสูง เกิดอนุภาคจำนวนมาก. "
        "โฟตอนส่วนใหญ่ในสถานะสุดท้ายมาจากการสลายของมีซอนเป็นกลาง โดยเฉพาะ "
        "\\(\\pi^0 \\to \\gamma\\gamma\\) (โอกาส ~98.8%) และ \\(\\eta \\to \\gamma\\gamma\\). "
        "นี่คือคำตอบของ \"สองโฟตอน\" — \\(\\pi^0\\) หนึ่งตัวแตกเป็นสองโฟตอน. "
        "<br>(ชุดข้อมูลของเราเป็นโฟตอนเดี่ยวที่ถูกจับคู่กับความจริงแบบ 1:1 เพื่อใช้ฝึกโมเดล).",
        "At LHCb two proton beams collide (pp collision) at high energy, producing many particles. "
        "Most final-state photons come from neutral-meson decays, especially "
        "\\(\\pi^0 \\to \\gamma\\gamma\\) (BR ~98.8%) and \\(\\eta \\to \\gamma\\gamma\\). "
        "That is the \"two photons\": one \\(\\pi^0\\) splitting into two photons. "
        "<br>(Our dataset is single photons truth-matched 1:1 for training.)",
    ),
    _fund(
        "how_travel", "2. เดินทาง & เรารู้โมเมนตัมได้ไง", "2. How it travels & how we know its momentum",
        "โฟตอนไม่มีประจุ จึงไม่ถูกสนามแม่เหล็ก dipole ของ LHCb ดัดทิศ — เดินทางเป็น <b>เส้นตรง</b> "
        "จากจุดกำเนิด (sig_flux_prod_vertex) ไปยังหน้า ECAL ที่ \\(z \\approx 12.62\\) m "
        "(อนุภาคมีประจุจะโค้งในสนาม ใช้ความโค้งวัดโมเมนตัม). "
        "<br><b>เรารู้โมเมนตัม \"เป๊ะ\" ได้เพราะเป็น simulation</b>: Geant4 กำหนด 4-momentum จริง "
        "(px, py, pz, E) ให้โฟตอนตั้งแต่ต้น แล้วเราเก็บเป็น truth (sig_flux_*). ในข้อมูลจริงเราไม่รู้ "
        "ต้องวัดเอา — จึงต้องฝึกโมเดลบน simulation.",
        "A photon is neutral, so the LHCb dipole magnet does not bend it — it travels in a "
        "<b>straight line</b> from its production vertex (sig_flux_prod_vertex) to the ECAL face at "
        "\\(z \\approx 12.62\\) m (charged particles curve in the field; the curvature measures their momentum). "
        "<br><b>We know the momentum exactly because this is simulation</b>: Geant4 assigns the photon a "
        "true 4-momentum (px, py, pz, E) up front, stored as truth (sig_flux_*). In real data we do not "
        "know it and must measure it — hence training on simulation.",
    ),
    _fund(
        "shower", "3. ชนเข้าเนื้อ detector แล้วเกิด shower", "3. Hitting the detector → an EM shower",
        "เมื่อโฟตอนพลังงานสูงเข้าวัสดุหนาแน่น มันเปลี่ยนเป็นคู่ \\(e^+e^-\\) (pair production); "
        "อิเล็กตรอน/โพซิตรอนเบรกในสนามนิวเคลียสแล้วแผ่โฟตอนใหม่ (bremsstrahlung); โฟตอนใหม่แตกคู่อีก … "
        "เป็น <b>ลูกโซ่ทวีคูณ (cascade)</b> จนพลังงานต่ำพอจะหยุด. นี่คือ \"ชนแล้วได้อนุภาคใหม่ ๆ\" ที่ถามถึง "
        "— คือ pair production + bremsstrahlung. ผลคือพลังงานหนึ่งโฟตอนกระจายไปหลายเซลล์.",
        "When a high-energy photon enters dense material it converts to an \\(e^+e^-\\) pair "
        "(pair production); the electron/positron brake in the nuclear field and radiate new photons "
        "(bremsstrahlung); those photons pair-produce again … a <b>multiplicative cascade</b> until the "
        "energy is too low to continue. This is the \"hitting and making new particles\" you asked about: "
        "pair production + bremsstrahlung. One photon's energy spreads across many cells.",
    ),
    _fund(
        "sampling", "4. Sampling calorimeter ทำงานยังไง", "4. How a sampling calorimeter works",
        "PicoCal เป็น <b>sampling calorimeter</b>: สลับชั้น <b>absorber</b> (วัสดุหนาแน่น — ทังสเตน W หรือ "
        "ตะกั่ว Pb — ทำให้เกิด shower) กับ <b>scintillator</b> (วัสดุเรืองแสง — เปล่งแสงตามพลังงานที่ทิ้งไว้). "
        "เราเก็บแสงได้แค่ <i>เศษส่วน</i> ของพลังงานทั้งหมด (sampling fraction) → เป็นที่มาของ stochastic term "
        "ใน energy resolution.",
        "PicoCal is a <b>sampling calorimeter</b>: alternating <b>absorber</b> (dense material — tungsten W "
        "or lead Pb — which makes the shower) and <b>scintillator</b> (which emits light proportional to the "
        "deposited energy). Only a <i>fraction</i> of the energy is sampled as light (the sampling fraction) "
        "— the source of the stochastic term in the energy resolution.",
    ),
    _fund(
        "technologies", "5. เทคโนโลยีโมดูลของ PicoCal", "5. PicoCal module technologies",
        "ตามบริเวณ detector (paper Fig.1–2):<br>"
        "• <b>SpaCal-W</b>: ทังสเตน + ผลึกเรืองแสง garnet (GAGG) ทนรังสี — กลาง detector ละเอียดสุด (1.5, 3 cm)<br>"
        "• <b>SpaCal-Pb</b>: ตะกั่ว + เส้นใยพลาสติกเรืองแสง (6 cm)<br>"
        "• <b>Shashlik</b>: แผ่นตะกั่วสลับ polystyrene + เส้นใย WLS — บริเวณนอก (4, 12 cm)<br>"
        "\"SpaCal\" = Spaghetti Calorimeter เพราะเส้นใยวางตามยาวเหมือนเส้นสปาเกตตี. [NIMA 1079 (2025) 170608]",
        "By detector region (paper Fig.1–2):<br>"
        "• <b>SpaCal-W</b>: tungsten + radiation-hard garnet (GAGG) crystals — innermost, finest (1.5, 3 cm)<br>"
        "• <b>SpaCal-Pb</b>: lead + plastic scintillating fibres (6 cm)<br>"
        "• <b>Shashlik</b>: lead tiles alternating with polystyrene + WLS fibres — outer (4, 12 cm)<br>"
        "\"SpaCal\" = Spaghetti Calorimeter, because the fibres run lengthwise like spaghetti. "
        "[NIMA 1079 (2025) 170608]",
    ),
    _fund(
        "readout", "6. วัดยังไง — ห่วงโซ่การอ่านสัญญาณ", "6. How it measures — the readout chain",
        "shower เปล่ง <b>แสง scintillation</b> → นำแสงผ่านเส้นใย/ท่อนำแสง (hollow light guide) หรือเส้นใย WLS "
        "→ <b>PMT (photomultiplier)</b> เปลี่ยนแสงเป็นสัญญาณไฟฟ้า → <b>ADC</b> อินทิเกรตประจุ (เกต 400 ns) "
        "= พลังงาน; <b>DRS4 digitizer</b> จับรูปคลื่น = เวลา. Run5 อ่านทั้งสองด้าน (double-sided) และแบ่งหน้า/หลัง "
        "→ เวลาแม่นขึ้นโดยรวม timestamp หน้า/หลังเพื่อแก้ jitter จาก shower fluctuation.",
        "The shower emits <b>scintillation light</b> → carried by fibres / hollow light guides or WLS fibres "
        "→ a <b>PMT (photomultiplier)</b> converts light to an electrical signal → an <b>ADC</b> integrates "
        "the charge (400 ns gate) = energy; a <b>DRS4 digitizer</b> captures the waveform = time. Run 5 reads "
        "both ends (double-sided) and splits front/back → better timing by combining front/back timestamps to "
        "correct shower-fluctuation jitter.",
    ),
    _fund(
        "reco_cluster", "จากเซลล์ดิบเป็น cluster (อัลกอริทึม reco)", "From raw cells to a cluster",
        "ขั้นตอน reconstruction: (1) หา <b>seed</b> = เซลล์พลังงานสูงสุดในท้องถิ่น (local maximum); "
        "(2) ดึงหน้าต่างรอบ seed (<b>3×3 หรือ 5×5 โมดูล</b>) มารวมเป็น cluster; "
        "(3) รวมพลังงานเซลล์ + หา centroid ถ่วงพลังงาน = ตำแหน่ง cluster. "
        "ชุดข้อมูลของเราคือ <i>ผลลัพธ์</i> ของขั้นนี้ (หน้าต่าง 5×5 รอบ seed, จับคู่กับโฟตอนจริง). "
        "อัลกอริทึม production ปัจจุบันของ LHCb (rule-based / graph-based) คือ "
        "<b>baseline ที่โปรเจกต์เราตั้งใจเอาชนะ</b>ด้วย space-time transformer.",
        "Reconstruction steps: (1) find a <b>seed</b> = a local-maximum (highest-energy) cell; "
        "(2) take a window around it (<b>3×3 or 5×5 modules</b>) as the cluster; "
        "(3) sum the cell energies + take the energy-weighted centroid = cluster position. "
        "Our dataset is the <i>output</i> of this step (a 5×5 window around the seed, matched to a true photon). "
        "LHCb's current production algorithm (rule-based / graph-based) is the "
        "<b>baseline this project aims to beat</b> with a space-time transformer.",
    ),
    _fund(
        "measure_what", "7. วัดอะไรบ้าง + ตัวเลขจริง", "7. What it measures + real numbers",
        "(1) <b>พลังงาน</b> — แสงรวม ∝ พลังงานที่ทิ้ง, คาลิเบรตเป็น MeV ต่อเซลล์ แล้วรวมเป็น cluster; "
        "resolution จริง \\(\\sigma_E/E \\approx 10\\%/\\sqrt{E}\\oplus 1\\%\\) (W/poly 9.9%/1.1%). "
        "<br>(2) <b>เวลา</b> — เวลาสัญญาณมาถึง (ns), แม่น < 20 ps ที่พลังงานสูง. "
        "<br>(3) <b>ตำแหน่ง</b> — เซลล์ไหนติด + centroid ถ่วงพลังงาน = (x_cluster, y_cluster). "
        "[NIMA 1079 (2025) 170608]",
        "(1) <b>Energy</b> — total light ∝ deposited energy, calibrated to MeV per cell, summed into a cluster; "
        "measured \\(\\sigma_E/E \\approx 10\\%/\\sqrt{E}\\oplus 1\\%\\) (W/poly 9.9%/1.1%). "
        "<br>(2) <b>Time</b> — signal arrival time (ns), better than 20 ps at high energy. "
        "<br>(3) <b>Position</b> — which cells fired + the energy-weighted centroid = (x_cluster, y_cluster). "
        "[NIMA 1079 (2025) 170608]",
    ),
    _fund(
        "timing_pileup", "8. ทำไมต้องวัดเวลาแม่นระดับ ps", "8. Why picosecond timing",
        "Upgrade II ลูมิโนซิตีสูงมาก เกิดการชนซ้อนกันหลายครั้งในเฟรมเดียว (<b>pile-up</b>). "
        "timestamp ระดับ ps ช่วยแยกว่าพลังงานก้อนนี้มาจากการชนครั้งไหน — ลดความสับสนจาก pile-up. "
        "นี่คือมิติ \"เวลา\" ที่ทำให้โมเดลของเราเป็น <b>spacetime</b> transformer (x, y, t).",
        "Upgrade II runs at very high luminosity, so many collisions overlap in one frame (<b>pile-up</b>). "
        "Picosecond timestamps tell which collision a deposit belongs to, cutting pile-up confusion. "
        "This is the \"time\" dimension that makes our model a <b>spacetime</b> transformer (x, y, t).",
    ),
    _fund(
        "why_upgrade", "9. ทำไมต้องสร้าง PicoCal", "9. Why build the PicoCal",
        "หลัง Run 3 ECAL เดิมเสียหายจากรังสี และ Upgrade II (ลูมิ ~\\(1.5\\times10^{34}\\,\\text{cm}^{-2}\\text{s}^{-1}\\), "
        "300 fb⁻¹) มี occupancy สูงขึ้นมาก จึงต้อง: วัสดุทนรังสี (garnet/tungsten), granularity ละเอียดขึ้น "
        "(1.5 cm กลาง), และ timing แม่นระดับ ps. [NIMA 1079 (2025) 170608]",
        "After Run 3 the old ECAL is radiation-damaged, and Upgrade II "
        "(\\(\\sim 1.5\\times10^{34}\\,\\text{cm}^{-2}\\text{s}^{-1}\\), 300 fb⁻¹) brings much higher occupancy, "
        "so it needs radiation-hard materials (garnet/tungsten), finer granularity (1.5 cm in the centre), and "
        "picosecond timing. [NIMA 1079 (2025) 170608]",
    ),
    _fund(
        "truth_vs_reco", "10. สรุปเส้นทางทั้งหมด → ทำไมต้อง ML", "10. The whole chain → why ML",
        "<b>pp → \\(\\pi^0\\to\\gamma\\gamma\\) → โฟตอนเดินทางตรงเข้าหน้า ECAL → EM shower → เซลล์เก็บแสง "
        "→ PMT → สัญญาณ → cluster.</b> ใน simulation เรารู้พลังงานจริง (truth); ในข้อมูลจริงไม่รู้ → "
        "ฝึกโมเดลบน sim เพื่อทำนายพลังงานจริงจากรูปแบบเซลล์ (อ่านต่อหน้า \"สำรวจ & สูตร\").",
        "<b>pp → \\(\\pi^0\\to\\gamma\\gamma\\) → photon travels straight to the ECAL face → EM shower → cells "
        "collect light → PMT → signal → cluster.</b> In simulation we know the true energy (truth); in real "
        "data we do not → train on simulation to predict the true energy from the cell pattern (continue in "
        "\"Exploration & Formulas\").",
    ),
]


def explainers():
    return {"plots": PLOTS, "formulas": FORMULAS, "fundamentals": FUNDAMENTALS}
