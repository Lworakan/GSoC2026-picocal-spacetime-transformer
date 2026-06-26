from picocal_explorer import dictionary


def test_dictionary_marks_derived_and_units():
    by = {f["name"]: f for f in dictionary.dictionary()}
    assert by["cell_pitch"]["present"] == "derived"
    assert by["sig_flux_eTot"]["unit"] == "GeV"
    assert by["energy"]["present"] == "root"


def test_live_schema_real_file():
    rows = {r["name"]: r for r in dictionary.live_schema("matched_1001_1010.root")}
    assert rows["energy"]["kind"] == "jagged"
    assert rows["sig_flux_eTot"]["kind"] == "scalar"
