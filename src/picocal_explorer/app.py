from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import analysis, data, dictionary, explainers

app = FastAPI(title="PicoCal Data Explorer")
STATIC = Path(__file__).resolve().parents[2] / "static"


@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/api/files")
def api_files():
    return data.list_files()


@app.get("/api/files/{name}/overview")
def api_overview(name: str):
    try:
        return data.file_overview(name)
    except FileNotFoundError:
        raise HTTPException(404, name)


@app.get("/api/files/{name}/event/{event_id}")
def api_event(name: str, event_id: int):
    try:
        return data.event_detail(name, event_id)
    except FileNotFoundError:
        raise HTTPException(404, name)
    except KeyError:
        raise HTTPException(404, f"event {event_id}")


@app.get("/api/files/{name}/distributions")
def api_distributions(name: str):
    try:
        return analysis.distributions(name)
    except FileNotFoundError:
        raise HTTPException(404, name)


@app.get("/api/files/{name}/schema")
def api_schema(name: str):
    try:
        return dictionary.live_schema(name)
    except FileNotFoundError:
        raise HTTPException(404, name)


@app.get("/api/dictionary")
def api_dictionary():
    return dictionary.dictionary()


@app.get("/api/explainers")
def api_explainers():
    return explainers.explainers()


@app.get("/api/geometry")
def api_geometry():
    mm = data.module_map()
    return data._safe([{"imodx": k[0], "jmody": k[1], **v} for k, v in mm.items()])


@app.on_event("startup")
def _warm():
    data.module_map()


if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
