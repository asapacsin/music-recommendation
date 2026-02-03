from fastapi import FastAPI
import app.recommend as recom
import os
import logging
from pathlib import Path

app = FastAPI()

@app.post("/recommend")
def api_recommend(file_path:str):
    # path = Path.cwd()
    get_file_path=os.path.join("data/input", file_path)
    get_recommend=recom.recommend(get_file_path)
    return {"recommend":str(get_recommend)}