import os

import torch
import librosa
import soundfile as sf
import torch.utils.data as tdata
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import numpy as np
import warnings
import h5py
warnings.filterwarnings('ignore')

class AudioDataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe.
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,data_list_path,num_classes=10,n_mels=64,n_fft=2048,hop_length=380,win_length=512,sr=16000,EPS = np.spacing(1),fre_len=100,mode='train'):
        super().__init__()
        self.MEL_ARGS = {
            'n_mels': n_mels,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length
        }
        self.dataset_path = []
        self.labels = []
        self.num_classes = num_classes
        with open(data_list_path) as f:
            data = f.readlines()
        for i in data:
            self.dataset_path.append(i.split()[0])
            self.labels.append(one_hot(torch.tensor(eval(i.split()[1])),num_classes=self.num_classes))
        self.sr = sr
        self.EPS = EPS
        self.fre_len = fre_len

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):

        wavpath = self.dataset_path[index]
        y, sr = sf.read(wavpath, dtype='float32')
        if y.ndim > 1:
            y = y.mean(1)
            y = librosa.resample(y, sr, self.sr)
        label = self.labels[index]
        data = torch.tensor(np.log(librosa.feature.melspectrogram(y, **self.MEL_ARGS) + self.EPS)).unsqueeze(dim=0)
        if data.shape[2] < self.fre_len:
            num = self.fre_len //data.shape[2] + 1
            data = data.repeat(1, 1, num)

        data = data[:,:,:self.fre_len]
        return data, label,wavpath

if __name__ == "__main__":

    for data_list in ["train","test"]:
        print(f"{data_list} process end")
        dataset = AudioDataset(data_list_path=f"dataset/{data_list}_list.txt")
        train_loader = DataLoader(dataset, batch_size=8, num_workers=0)
        h5py_path = f"features/Urbansound8K_{data_list}.h5"
        os.makedirs(os.path.dirname(h5py_path),exist_ok=True)
        with h5py.File(h5py_path, 'w') as store:
            for dates,labels,paths in train_loader:
                for date, label, path_name in zip(dates, labels, paths):
                    path_name = path_name.split("/")
                    path_name = "_".join(path_name[-4:])
                    store[path_name + "_data"] = date
                    store[path_name + "_label"] = label

        print(f"{data_list} process end")