from fastapi import FastAPI
from pydantic import BaseModel
from ml import search, result

app = FastAPI()

@app.get("/")
def read_root():
    return {'Hello': 'World'}

class Song(BaseModel):
    name: str
    index: int
    artist: str

@app.post("/songs/")
def create_songs(song: Song):
    return song

@app.get("/search/")
def search_song(prompt:str):
    pred_id = search()
    song = result(pred_id)
    return song