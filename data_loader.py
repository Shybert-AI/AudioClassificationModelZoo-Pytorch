import math
import torch
import librosa
import soundfile as sf
import torch.utils.data as tdata
from torch.nn.functional import one_hot
import torch.nn.functional as F
import numpy as np
from h5py import File
import warnings
warnings.filterwarnings('ignore')

class AudioDataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,data_list_path,num_classes=10,transform=None,n_mels=64,n_fft=2048,hop_length=380,win_length=512,sr=16000,EPS = np.spacing(1),fre_len=100,mode='train'):
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
        self._transform = transform

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
        if self._transform:
            data = self._transform(data)
        return data, label,wavpath


class AudioDataset_test(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe.
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,data_list_path,num_classes=10,transform=None,n_mels=64,n_fft=2048,hop_length=380,win_length=512,sr=16000,EPS = np.spacing(1),fre_len=100,audio_path="example_audio/7383-3-0-1.wav"):
        super().__init__()
        self.MEL_ARGS = {
            'n_mels': n_mels,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': win_length
        }
        self.dataset_path = [audio_path]
        self.labels = []
        self.num_classes = num_classes
        self.sr = sr
        self.EPS = EPS
        self.fre_len = fre_len
        self._transform = transform

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):

        wavpath = self.dataset_path[index]
        y, sr = sf.read(wavpath, dtype='float32')
        if y.ndim > 1:
            y = y.mean(1)
            y = librosa.resample(y, sr, self.sr)

        data = torch.tensor(np.log(librosa.feature.melspectrogram(y, **self.MEL_ARGS) + self.EPS)).unsqueeze(dim=0)
        num = 0
        if data.shape[2]%self.fre_len !=0:
            num = math.ceil(data.shape[2] / self.fre_len)
            padding = self.fre_len*num - data.shape[2]
            data = F.pad(data, (0, padding, 0, 0, 0, 0))

        return data,wavpath,num

class AudioDataset_Feature(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe.
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,data_list_path,feature_map_list,num_classes=10,transform=None,mode='train'):
        super().__init__()
        self.dataset_path = []
        self.num_classes = num_classes
        self._h5file = feature_map_list
        with open(data_list_path) as f:
            data = f.readlines()
        for i in data:
            self.dataset_path.append(i.split()[0])
        self.dataset = File(self._h5file, 'r', libver='latest')
        self._transform = transform

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, index):
        wavpath = self.dataset_path[index]
        path_name = wavpath.split("/")
        path_name = "_".join(path_name[-4:])
        data = self.dataset[path_name+"_data"][()]
        label = self.dataset[path_name+"_label"][()]
        if self._transform:
            data = self._transform(data)
        return torch.tensor(data), torch.tensor(label),wavpath

if __name__ == "__main__":
    dataset = AudioDataset(data_list_path="dataset/test_list.txt")
    for date,label,path in dataset:
        print(date.shape)
        print(label)
        print(type(date))
        print(type(label))
        print(path)
        break

    dataset = AudioDataset_Feature(data_list_path="dataset/train_list.txt",feature_map_list="features/Urbansound8K_train.h5")
    for date,label,path in dataset:
        print(date)
        print(label)
        print(type(date))
        print(type(label))
        print(path)
        break