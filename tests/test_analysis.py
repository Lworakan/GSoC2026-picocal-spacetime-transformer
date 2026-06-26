from picocal_explorer import analysis


def test_distributions_shapes():
    d = analysis.distributions("matched_1001_1010.root")
    assert len(d["truth_energy_gev"]) == d["meta"]["n_entries"]
    assert len(d["response"]) == d["meta"]["n_entries"]
    assert all(v < 1000 for v in d["truth_energy_gev"])
