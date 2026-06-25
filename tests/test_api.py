from fastapi.testclient import TestClient

from picocal_explorer.app import app

client = TestClient(app)


def test_files_endpoint():
    r = client.get("/api/files")
    assert r.status_code == 200 and len(r.json()) >= 100


def test_event_endpoint():
    r = client.get("/api/files/matched_1001_1010.root/event/4")
    j = r.json()
    assert len(j["truth_photons"]) == 3 and len(j["clusters"]) == 1


def test_explainers_endpoint():
    assert client.get("/api/explainers").status_code == 200


def test_geometry_endpoint():
    r = client.get("/api/geometry")
    assert r.status_code == 200 and len(r.json()) > 0


def test_distributions_endpoint():
    r = client.get("/api/files/matched_1001_1010.root/distributions")
    assert r.status_code == 200 and r.json()["meta"]["n_entries"] > 0
