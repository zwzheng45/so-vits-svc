import math
import multiprocessing
import os
import argparse
from random import shuffle

import torch
import json
from glob import glob

from pyworld import pyworld
from tqdm import tqdm
from scipy.io import wavfile

import utils
from modules.mel_processing import mel_spectrogram_torch
#import h5py
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import parselmouth
import librosa
import numpy as np

hps = utils.get_hparams_from_file("configs/config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length

def get_f0(path,p_len=None, f0_up_key=0):
    x, _ = librosa.load(path, sampling_rate)
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 3, (path, p_len, x.shape)
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0bak = f0.copy()
    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0bak

def resize2d(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def compute_f0(wav):
    f0, t = pyworld.dio(
        wav.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(wav.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return f0




def process_one(filename, hmodel):
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).unsqueeze(0).to(devive)
        c = utils.get_hubert_content(hmodel, wav16k, device="cuda" if torch.cuda.is_available() else "cpu")
        torch.save(c.cpu(), soft_path)
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = compute_f0(wav)
        np.save(f0_path, f0)


def process_batch(filenames):
    print("Loading hubert for content...")
    hmodel = utils.get_hubert_model(0 if torch.cuda.is_available() else None)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, hmodel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="dataset/44k", help="path to input dir")

    args = parser.parse_args()
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method('spawn')

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]
    print([len(c) for c in chunks])
    processes = [multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks]
    for p in processes:
        p.start()
