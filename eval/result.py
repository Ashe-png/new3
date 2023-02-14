import pandas as pd

def result(pred_id):

    df = pd.read_json('./eval/metadata.json')

    indexes = df.iloc[:,1].values
    closest = df.iloc[(df['index']-pred_id).abs().argsort()[:2]]

    song = closest[closest['index']<=pred_id].values
    print(f'The song name is {song[0][0]}')


