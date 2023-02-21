import pandas as pd
import json

def result(pred_id):

    with open('./eval/metadata.json', 'r') as f:
        data = json.load(f)
    column_data = [row['indices'] for row in data]

    filtered_list = [x for x in column_data if x <= pred_id]
    closest = min(filtered_list, key=lambda x: abs(x - pred_id))

    for song in data:
        if song['indices'] == closest:
            print (song)
            return song
        


