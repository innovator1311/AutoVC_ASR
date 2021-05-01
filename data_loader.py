from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, deepspeech_dir, len_crop):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.deepspeech_dir = deepspeech_dir
	self.len_crop = len_crop
        self.step = 10
        
        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = [os.path.join(self.root_dir, tmp), os.path.join(self.deepspeech_dir, tmp)]
            dataset[idx_offset+k] = uttrs
                   
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        
	tmp = list_uttrs[a]
	tmp_mel = np.load(tmp[0])
	tmp_deepspeech = np.load(tmp[1])

        if tmp_mel.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp_mel.shape[0]
	    len_ds_pad = self.len_crop // 2 - tmp_deepspeech.shape[0]
            uttr = np.pad(tmp_mel, ((0,len_pad),(0,0)), 'constant')
	    deepspeech = np.pad(tmp_deepspeech, ((0,len_ds_pad),(0,0)), 'constant') 
        elif tmp_mel.shape[0] > self.len_crop:
            left = np.random.randint(tmp_mel.shape[0]-self.len_crop)
            uttr = tmp_mel[left:left+self.len_crop, :]
	    dp_left = left//2
	    deepspeech = tmp_deepspeech[dp_left:dp_left + self.len_crop//2, :]
        else:
            uttr = tmp_mel
	    deepspeech = tmp_deepspeech

        return uttr, deepspeech ,emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(root_dir, batch_size=16, len_crop=128, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, len_crop)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






