import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator

import soundfile as sf

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)

G = torch.load('checkpoints/checkpoint_1000.pt',map_location=torch.device('cuda'))

metadata = pickle.load(open('metadata.pkl', "rb"))

spect_vc = []

for sbmt_i in metadata:
         
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)

import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("../drive/MyDrive/MultiSpeaker_Tacotron2/checkpoint_step001000000_ema.pth",map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    #librosa.output.write_wav(name+'.wav', waveform, sr=16000)
    sf.write(name+'.wav', waveform, 16000,subtype='PCM_24')