from picocal_explorer import explainers


def test_explainers_bilingual_and_ids():
    e = explainers.explainers()
    ids = {p["id"] for p in e["plots"]} | {f["id"] for f in e["formulas"]}
    assert {"response", "truth_spectrum", "resolution", "why_transformer"} <= ids
    r = next(f for f in e["formulas"] if f["id"] == "resolution")
    assert r["formula_latex"] and r["what"]["th"] and r["what"]["en"]
