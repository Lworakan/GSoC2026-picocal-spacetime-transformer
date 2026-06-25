from picocal_explorer import data


def test_list_files_has_full():
    files = data.list_files()
    assert any(f["name"] == "matched_1001_1010.root" for f in files)


def test_event_detail_event4_three_photons_one_cluster():
    d = data.event_detail("matched_1001_1010.root", 4)
    assert len(d["truth_photons"]) == 3
    assert len(d["clusters"]) == 1
    c = d["clusters"][0]
    assert c["n_cells"] == len(c["cells"])
    assert any(cell["is_seed"] for cell in c["cells"])
    assert "pitch_derived" in c["cells"][0]


def test_truth_energy_is_gev():
    d = data.event_detail("matched_1001_1010.root", 4)
    assert max(p["energy_gev"] for p in d["truth_photons"]) < 1000
