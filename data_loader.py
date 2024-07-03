import logging
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
__all__ = ['MMDataLoader']

logger = logging.getLogger('MMSA')




class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.with_control = self.args['with_control']
        DATASET_MAP = {
            'attractiveness': self.__init_attractiveness,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_attractiveness(self):

        print("featurePath:\n",self.args['featurePath'])
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
            f.close()
        print(self.mode)

        self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        try:
           self.info = data[self.mode]['info'].tolist() #(2679, 'XJ52666888$_$379') (2303, 'suyunee$_$571') (473, '91647030$_$802') (2640, 'XJ52666888$_$688') (1527, 'GRNtongxie$_$107') (270, '28_12_20$_$941') (111, '1444972715$_$266') (2248, 'SN851108$_$218') (130, '28_12_20$_$850') (2508, 'XJ52666888$_$298') (
        except:
            self.info = data[self.mode]['info']
        self.raw_text = data[self.mode]['raw_text'].tolist()
        self.labels = {'M': np.array(data[self.mode]['labels']).astype(np.float32)}
        self.ProductType_experience =   data[self.mode]['product_ClothesBeatuty']

        # remove nan
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0

        if  self.with_control:
            self.control = [ v for v in self.ProductType_experience]

        print("!!! type of data is np.array" )
        self.args['feature_dims'][0] = self.text.shape[2]
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.args['feature_dims'][2] = self.vision.shape[2]


        logger.info(f"{self.mode} samples: {self.text[0].shape}")

        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return  self.text.shape[0]

    def get_seq_len(self):
       return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):

        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'raw_text': self.raw_text[index],
            "meta_info": self.info[index]
        }

        if self.with_control:
            t_control = self.control[index]
            tensor_control = torch.tensor(t_control, dtype=torch.float32)
            sample['control'] = tensor_control

        sample['audio_lengths'] = self.audio.shape[0]
        sample['vision_lengths'] = self.vision.shape[0]
        return sample


def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len()
        # todo
        print("args['seq_lens'] :",args['seq_lens'] ) # todo

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=8,
                       num_workers=0,
                       shuffle=False)
        for ds in datasets.keys()
    }

    return dataLoader
