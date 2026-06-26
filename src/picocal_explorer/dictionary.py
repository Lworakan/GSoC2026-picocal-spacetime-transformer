from __future__ import annotations

import awkward as ak
import numpy as np
import uproot


def _f(name, level, kind, unit, present, what_th, what_en, formula="", note_th="", note_en=""):
    return {
        "name": name,
        "level": level,
        "kind": kind,
        "unit": unit,
        "formula": formula,
        "present": present,
        "source": "ROOT" if present == "root" else "derived",
        "text": {"what": {"th": what_th, "en": what_en}},
        "note": {"th": note_th, "en": note_en},
    }


FIELDS = [
    _f("event", "event", "scalar", "", "root",
       "เลข event เดิมใน TTree ใช้เป็นตัวจัดกลุ่ม (หนึ่ง event มีได้หลาย entry)",
       "Original TTree event index; a grouping id (one event can span several entries)",
       note_th="instruction บอก 1 event = 1 cluster แต่จริงมีหลาย entry ต่อ event",
       note_en="instruction says 1 event = 1 cluster, but real files have several entries per event"),
    _f("sig_flux_eTot", "truth", "scalar", "GeV", "root",
       "พลังงานจริงของโฟตอน — เป้าหมายของการ regression",
       "True photon energy — the regression target",
       note_th="instruction เขียน MeV แต่จริงเป็น GeV (×1000 ก่อนเทียบกับ total_energy)",
       note_en="instruction says MeV but it is GeV (×1000 before comparing to total_energy)"),
    _f("sig_flux_pdgID", "truth", "scalar", "", "root",
       "รหัสชนิดอนุภาค (22 = โฟตอน)", "Particle type code (22 = photon)"),
    _f("sig_flux_entry_x", "truth", "scalar", "mm", "root",
       "ตำแหน่ง x ที่อนุภาคจริงเข้าหน้า ECAL (จุดอ้างอิงสำหรับ matching)",
       "x where the true particle enters the ECAL face (matching reference)"),
    _f("sig_flux_entry_y", "truth", "scalar", "mm", "root",
       "ตำแหน่ง y ที่อนุภาคจริงเข้าหน้า ECAL", "y where the true particle enters the ECAL face"),
    _f("sig_flux_entry_z", "truth", "scalar", "mm", "root",
       "ตำแหน่ง z ของหน้า ECAL (คงที่ ~12.62 m)", "z of the ECAL face (constant ~12.62 m)"),
    _f("sig_flux_px", "truth", "scalar", "MeV/c", "root", "โมเมนตัมจริงแกน x", "true x momentum"),
    _f("sig_flux_py", "truth", "scalar", "MeV/c", "root", "โมเมนตัมจริงแกน y", "true y momentum"),
    _f("sig_flux_pz", "truth", "scalar", "MeV/c", "root",
       "โมเมนตัมจริงแกน z (ตามลำบีม, ค่าใหญ่สุด)", "true z momentum (along beam, dominant)"),
    _f("sig_flux_prod_vertex_x", "truth", "scalar", "mm", "root",
       "จุดกำเนิดอนุภาค x", "production vertex x"),
    _f("sig_flux_prod_vertex_y", "truth", "scalar", "mm", "root",
       "จุดกำเนิดอนุภาค y", "production vertex y"),
    _f("sig_flux_prod_vertex_z", "truth", "scalar", "mm", "root",
       "จุดกำเนิดอนุภาค z (ตำแหน่งตามลำบีมที่โฟตอนเกิด)",
       "production vertex z (where along the beamline the photon was created)"),
    _f("sig_flux_timing", "truth", "scalar", "ns", "root",
       "เวลาจริงของอนุภาค", "true particle timing"),
    _f("sig_dr_matched", "truth", "scalar", "mm", "root",
       "ระยะระหว่าง cluster ที่ reco กับจุดเข้าจริง (เล็ก = match ดี)",
       "distance between reco cluster and true entry (small = good match)"),
    _f("sig_dxdz_flux", "truth", "scalar", "", "root",
       "ทิศอนุภาค dx/dz = px/pz", "particle direction dx/dz = px/pz", formula="px / pz"),
    _f("sig_dydz_flux", "truth", "scalar", "", "root",
       "ทิศอนุภาค dy/dz = py/pz", "particle direction dy/dz = py/pz", formula="py / pz"),
    _f("x_cluster", "cluster", "scalar", "mm", "root",
       "ตำแหน่ง x ของ cluster ที่ reconstruct", "reconstructed cluster x position"),
    _f("y_cluster", "cluster", "scalar", "mm", "root",
       "ตำแหน่ง y ของ cluster ที่ reconstruct", "reconstructed cluster y position"),
    _f("total_energy", "cluster", "scalar", "MeV", "root",
       "พลังงานรวมที่ reco ของ cluster = front + back",
       "reconstructed total cluster energy = front + back", formula="total_energy_front + total_energy_back"),
    _f("total_energy_front", "cluster", "scalar", "MeV", "root",
       "พลังงานรวมส่วนหน้า", "total front-section energy"),
    _f("total_energy_back", "cluster", "scalar", "MeV", "root",
       "พลังงานรวมส่วนหลัง", "total back-section energy"),
    _f("seed_icell_x", "cluster", "scalar", "", "root",
       "ดัชนี x ของเซลล์ seed ในโมดูล", "x cell index of the seed within its module"),
    _f("seed_icell_y", "cluster", "scalar", "", "root",
       "ดัชนี y ของเซลล์ seed ในโมดูล", "y cell index of the seed within its module"),
    _f("imodx", "cell", "jagged", "", "root",
       "ดัชนีโมดูลแกน x (0 = ซ้ายสุด)", "module x-index (0 = left-most)"),
    _f("jmody", "cell", "jagged", "", "root",
       "ดัชนีโมดูลแกน y (0 = ล่างสุด)", "module y-index (0 = bottom-most)"),
    _f("icell", "cell", "jagged", "", "root",
       "ดัชนีเซลล์ภายในโมดูล", "1D cell index inside the module"),
    _f("cell_x", "cell", "jagged", "mm", "root",
       "พิกัด x ของเซลล์ (global)", "global cell x coordinate"),
    _f("cell_y", "cell", "jagged", "mm", "root",
       "พิกัด y ของเซลล์ (global)", "global cell y coordinate"),
    _f("cell_energies_front", "cell", "jagged", "MeV", "root",
       "พลังงานที่เซลล์เก็บส่วนหน้า", "energy deposited in the front segment"),
    _f("cell_energies_back", "cell", "jagged", "MeV", "root",
       "พลังงานที่เซลล์เก็บส่วนหลัง", "energy deposited in the back segment"),
    _f("energy", "cell", "jagged", "MeV", "root",
       "พลังงานรวมต่อเซลล์ = front + back (หนึ่ง token ต่อหนึ่งเซลล์)",
       "total cell energy = front + back (one token per cell)", formula="front + back"),
    _f("cell_times_front", "cell", "jagged", "ns", "root",
       "เวลาส่วนหน้าของเซลล์", "front-segment time"),
    _f("cell_times_back", "cell", "jagged", "ns", "root",
       "เวลาส่วนหลังของเซลล์", "back-segment time"),
    _f("seed_cell_x", "cell", "jagged", "mm", "root",
       "พิกัด x ของหน้าต่าง 3×3 รอบ seed", "x of the 3×3 window around the seed"),
    _f("seed_cell_y", "cell", "jagged", "mm", "root",
       "พิกัด y ของหน้าต่าง 3×3 รอบ seed", "y of the 3×3 window around the seed"),
    _f("cell_pitch", "cell", "derived", "mm", "derived",
       "ขนาดเซลล์ (อนุมานจากระยะเพื่อนบ้านในโมดูล)",
       "cell size (derived from neighbour spacing within the module)",
       formula="median nearest-neighbour distance",
       note_th="ไม่ได้เก็บในไฟล์ ROOT — คำนวณขึ้นมา", note_en="not stored in ROOT — computed"),
    _f("cell_modType", "cell", "derived", "", "derived",
       "ชนิดโมดูลจากขนาดเซลล์ (ตามรูป Fig.2 ของ paper): 1.5/3cm=SpaCal-W, 4/12cm=Shashlik, 6cm=SpaCal-Pb",
       "module type from cell size (per paper Fig.2): 1.5/3cm=SpaCal-W, 4/12cm=Shashlik, 6cm=SpaCal-Pb",
       note_th="ไม่ได้เก็บในไฟล์ ROOT — คำนวณจาก pitch. instruction ใช้ชื่ออีกแบบ (N-cell) ที่ไม่ตรงกับ Fig.2 ทั้งหมด",
       note_en="not stored in ROOT — derived from pitch. The instruction uses a different N-cell naming that does not fully match Fig.2"),
    _f("cell_rel_x", "cell", "derived", "mm", "derived",
       "ตำแหน่ง x ของเซลล์เทียบ seed", "cell x relative to the seed",
       formula="x_cell - x_seed",
       note_th="ไม่ได้เก็บในไฟล์ ROOT — คำนวณขึ้นมา", note_en="not stored in ROOT — computed"),
    _f("cell_rel_y", "cell", "derived", "mm", "derived",
       "ตำแหน่ง y ของเซลล์เทียบ seed", "cell y relative to the seed",
       formula="y_cell - y_seed",
       note_th="ไม่ได้เก็บในไฟล์ ROOT — คำนวณขึ้นมา", note_en="not stored in ROOT — computed"),
    _f("cell_rel_dr", "cell", "derived", "mm", "derived",
       "ระยะเซลล์ถึง seed", "radial distance from cell to seed",
       formula="sqrt(rel_x^2 + rel_y^2)",
       note_th="ไม่ได้เก็บในไฟล์ ROOT — คำนวณขึ้นมา", note_en="not stored in ROOT — computed"),
]


def dictionary():
    return FIELDS


def live_schema(name):
    from .data import TREE, _find

    by = {f["name"]: f for f in FIELDS}
    rows = []
    with uproot.open(_find(name)) as f:
        t = f[TREE]
        for n in t.keys():
            tn = t[n].typename
            jag = tn.endswith("[]")
            smin = smax = None
            if not jag:
                v = ak.to_numpy(t[n].array(library="ak"))
                if v.dtype.kind == "f":
                    v = v[np.isfinite(v)]
                if len(v):
                    smin, smax = float(np.min(v)), float(np.max(v))
            rows.append(
                {
                    "name": n,
                    "typename": tn,
                    "kind": "jagged" if jag else "scalar",
                    "sample_min": smin,
                    "sample_max": smax,
                    "meta": by.get(n, {}),
                }
            )
    return rows
