import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils.utils import read_wav_np


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)

def create_dataset(hp, args, train):
    dataset = MelFromDisk(hp, args, train)
    return dataset

class MelFromDisk(Dataset):
    def __init__(self, hp, meta_file, train):
        self.emotion_dict = {
                        'ang': 0,
                        'dis': 1,
                        'hap': 2,
                        'sad': 3,
                        'sca': 4,
                        'sur': 5,
                        'neu': 6
                    }
        self.hp = hp
        self.meta_file = meta_file
        self.meta_data = pd.read_csv(self.meta_file).to_numpy()
        self.train = train
        self.path = hp.data.train if train else hp.data.validation
        #self.wav_list = glob.glob(os.path.join(self.path, '**', '*.mel'), recursive=True)
        #self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length + 2
        #self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        #if self.train:
        emotion = self.emotion_dict[self.meta_data[idx][1]]
        emotion = torch.tensor(emotion)
        return self.my_getitem(idx), emotion

    #def shuffle_mapping(self):
        #random.shuffle(self.mapping)

    def my_getitem(self, idx):
        #wavpath = 
        #wavpath = self.wav_list[idx]
        melpath = self.path + self.meta_data[idx][0] + '.mel'
        #sr, audio = read_wav_np(wavpath)
        #if len(audio) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            #audio = np.pad(audio, (0, self.hp.audio.segment_length + self.hp.audio.pad_short - len(audio)), \
                    #mode='constant', constant_values=0.0)

        #audio = torch.from_numpy(audio).unsqueeze(0)
        mel = torch.load(melpath).squeeze(0).permute(1, 0)

        #if self.train:
            #max_mel_start = mel.size(1) - self.mel_segment_length
            #mel_start = random.randint(0, max_mel_start)
            #mel_end = mel_start + self.mel_segment_length
            #mel = mel[:, mel_start:mel_end]

            #audio_start = mel_start * self.hp.audio.hop_length
            #audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        #audio = audio + (1/32768) * torch.randn_like(audio)
        return mel
