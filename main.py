from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ml import search, result, load_memmap_data,get_index
import subprocess
from fastapi.responses import Response
import time

app = FastAPI()

emb_dir = './logs/emb/640_lamb/11/'
db,db_shape = load_memmap_data(emb_dir, 'custom_source')
# Create and train FAISS index
index_type = 'ivfpq'
nogpu = True
max_train = 1e7

index = get_index(index_type, db, db.shape, (not nogpu),
                    max_train, trained=True)


# Add items to index
start_time = time.time()

# index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
index.add(db); print(f'{len(db)} items from reference DB')

t = time.time() - start_time
print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')


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
    pred_id = search(db,db_shape,index)
    song = result(pred_id)
    return song

@app.post("/upload-audio")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
  
    return {"filename": file.filename}


# yo chai, maathi ko api mai integrate garna garho bhayo , so chuttei chuttei garna parcha.
# .aac bhanne file upload garna parcha, .wav bhanne file maa conver garna chai tala ko function.
def convertToWav():
    # get filename from previous function
    input_file = filename
    output_file = "output.wav"

    command = ["ffmpeg", "-i", input_file, "-ar", "18000", "-ac", "1", "-f", "wav", output_file]
    subprocess.run(command)
