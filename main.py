from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import subprocess
from fastapi.responses import Response
import time
import re
from pydub import AudioSegment
import ast
import os

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
    # song = search2('640_lamb', '11', 'default' ,'ivfpq', None, False, db, db_shape,index)
    # # song = result(pred_id)
    # return song
    result = subprocess.run(['python', 'run.py','search', '640_lamb', '11'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    pattern = r'\{.*?\}'

    # Find all matches of the pattern in the output
    match = re.findall(pattern, output)

    dict_string = match[0].strip("'")

    # Parse the dictionary string using the ast module
    parsed_dict = ast.literal_eval(dict_string)



    # Print the JSON object
    return parsed_dict

@app.post("/upload-audio")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()
    with open(f'temp\{file.filename}', "wb") as f:
        f.write(contents)
    
    command = ["ffmpeg", "-i", f'temp\{file.filename}', "-ar", "8000", "-ac", "1", "-f", "wav", f'temp\output.wav']
    subprocess.run(command)
    # audio = AudioSegment.from_file(f'temp\Antibiotic.m4a', format="m4a")

    # # Save the audio file as WAV
    # audio.export("temp\output_file.wav", format="wav")
  
    result = subprocess.run(['python', 'run.py','search', '640_lamb', '11'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    pattern = r'\{.*?\}'

    # Find all matches of the pattern in the output
    match = re.findall(pattern, output)

    dict_string = match[0].strip("'")

    # Parse the dictionary string using the ast module
    parsed_dict = ast.literal_eval(dict_string)

    os.remove('temp\output.wav')

    # Print the JSON object
    return parsed_dict


# yo chai, maathi ko api mai integrate garna garho bhayo , so chuttei chuttei garna parcha.
# .aac bhanne file upload garna parcha, .wav bhanne file maa conver garna chai tala ko function.
def convertToWav():
    # get filename from previous function
    input_file = filename
    output_file = "output.wav"
    

    command = ["ffmpeg", "-i", input_file, "-ar", "18000", "-ac", "1", "-f", "wav", output_file]
    subprocess.run(command)
