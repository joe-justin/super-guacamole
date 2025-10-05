# backend/main.py
import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

DATA_DIR = os.environ.get("DATA_DIR", "./output")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/functions")
def get_functions():
    fp = os.path.join(DATA_DIR, "functions.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="functions.json not found")
    return JSONResponse(content=__load_json(fp))

@app.get("/api/applications")
def get_applications():
    fp = os.path.join(DATA_DIR, "applications.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="applications.json not found")
    return JSONResponse(content=__load_json(fp))

@app.get("/api/servers")
def get_servers_all():
    fp = os.path.join(DATA_DIR, "servers_all.json")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="servers_all.json not found")
    return JSONResponse(content=__load_json(fp))

@app.get("/api/servers/by_application/{application}")
def get_servers_by_application(application: str):
    allp = os.path.join(DATA_DIR, "servers_all.json")
    if not os.path.exists(allp):
        raise HTTPException(status_code=404, detail="servers data missing")
    arr = __load_json(allp)
    filtered = [x for x in arr if x.get("Application") == application]
    return JSONResponse(content=filtered)

@app.get("/api/predictions/csv/{server}")
def get_server_csv(server: str):
    p = os.path.join(DATA_DIR, f"{server}_predicted.csv")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="csv not found")
    return FileResponse(p, media_type="text/csv", filename=f"{server}_predicted.csv")

@app.get("/api/server/{server}")
def get_server_info(server: str):
    p = os.path.join(DATA_DIR, f"{server}.pkl")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="server not found")
    with open(p, "rb") as f:
        info = pickle.load(f)
    return JSONResponse(content=info)

@app.get("/api/server/plot/{server}")
def api_server_plot(server: str):
    """
    Returns combined historical + predicted data for plotting.
    """
    csv_path = DATA_DIR / f"{server}_predicted.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"{csv_path.name} not found")

    df = pd.read_csv(csv_path)
    # Expect columns: timestamp, cpu_pred, mem_pred, cpu_actual, mem_actual ...
    # Ensure timestamps are sorted and serializable
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # convert to JSON rows for frontend plotting
    return JSONResponse(
        content=df.to_dict(orient="records")
    )


def __load_json(path):
    import json
    with open(path, "r") as f:
        return json.load(f)
