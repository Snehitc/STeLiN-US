# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:26:54 2024

@author: Snehit
"""

import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
from torch.utils.data import Dataset


def get_melspectrogram_db(file_path, Clip, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  wav = wav[int(Clip[0]*(sr/1000)):int(Clip[1]*(sr/1000))]
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db


def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled



class STeLiN_US_Events(Dataset):
  def __init__(self, disc, df, in_col1, in_col2, out_col, Events):
    self.df = df
    self.data = []
    self.labels = []
    
    for ind in tqdm(range(len(df))):
      row = df.iloc[ind]
      file_path = os.path.join(disc, "Audio", row[in_col1].split('_')[1], row[in_col1])
      Clip = row[in_col2]
      self.data.append(spec_to_image(get_melspectrogram_db(file_path, Clip))[np.newaxis,...])
      
      lab_e = np.zeros((len(Events)))
      for e in row[out_col]:
          lab_e[Events.index(e)] = 1
      self.labels.append(lab_e)
      
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

