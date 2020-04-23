import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import math
import librosa
import json
import wave
import sys
import pickle
import sklearn

import urllib.request
import librosa.display

from scipy.io import wavfile

import librosa.display
import soundfile as sf

import os
from tqdm import tqdm
import random
from pathlib import Path

audio_data = "Audio/"
path_preproccess = 'Audio_preprocess/'
annotations_data_by_emotions = "Annotations_by_emotions/"

def extractVectors(labels_df, sr = 44100):
    audio_vectors = {}
    for sess in [1]:  # using one session due to memory constraint, can replace [5] with range(1, 6)
        wav_file_path = audio_data
        orig_wav_files = os.listdir(wav_file_path)
        index = 0
        for orig_wav_file in tqdm(orig_wav_files):
            try:
                orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
                orig_wav_file, file_format = orig_wav_file.split('.')
                if orig_wav_file == '10dec_O11_1_mic':
                    continue
                for i, row in labels_df[labels_df['File'].str.contains(orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion = row['Start'], row['End'], row['File'], row['Emotion']
                    if end_time - start_time < 1.5:
                        continue
                    start_frame = math.floor(np.abs(start_time) * sr)
                    end_frame = math.floor(end_time * sr)
                    truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                    audio_vectors["{:05d}_".format(index) + truncated_wav_file_name] = (truncated_wav_vector, emotion)
                    index += 1
            except KeyboardInterrupt:
                break
            except:
                print('An exception occured for {}'.format(orig_wav_file))
                raise
    return audio_vectors

def writeVectors(audio_vectors):
    # sf.write('stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24')
    files = []
    emotions = []
    for i in tqdm(audio_vectors):
        files.append(path_preproccess + i)
        emotions.append(audio_vectors[i][1])
        sf.write(path_preproccess + i, audio_vectors[i][0], 44100, 'PCM_24')
    preprop_data = {"File": files, "Emotion" : emotions}
    df_prep = pd.DataFrame.from_dict(preprop_data)
    return df_prep

if __name__ == "__main__": 

    df_angry = pd.read_csv(annotations_data_by_emotions + 'data_Angry.csv', index_col='ID')
    df_disgusted = pd.read_csv(annotations_data_by_emotions + 'data_Disgusted.csv', index_col='ID')
    df_happy = pd.read_csv(annotations_data_by_emotions + 'data_Happy.csv', index_col='ID')
    df_neutral = pd.read_csv(annotations_data_by_emotions + 'data_Neutral.csv', index_col='ID')
    df_sad = pd.read_csv(annotations_data_by_emotions + 'data_Sad.csv', index_col='ID')
    df_scared = pd.read_csv(annotations_data_by_emotions + 'data_Scared.csv', index_col='ID')
    df_surprised = pd.read_csv(annotations_data_by_emotions + 'data_Surprised.csv', index_col='ID')

    df_angry['Emotion'] = 'ang'
    df_disgusted['Emotion'] = 'dis' 
    df_happy['Emotion'] = 'hap' 
    df_neutral['Emotion'] = 'neu' 
    df_sad['Emotion'] = 'sad'
    df_scared['Emotion'] = 'sca' 
    df_surprised['Emotion'] = 'sur'

    labels_df = pd.concat([df_happy, df_angry, df_disgusted, df_neutral, df_sad, df_scared, df_surprised])
    labels_df.File += '_mic.wav'
    
    Path(path_preproccess).mkdir(parents=True, exist_ok=True)
    audio_vectors = extractVectors(labels_df)
    df_prep = writeVectors(audio_vectors)
    df_prep.to_csv('df_prep.csv', index=False)