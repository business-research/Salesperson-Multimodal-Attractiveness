"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .FeatureNets import SubNet, TextSubNet


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, args):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.output_dim = args.num_classes if args.train_mode == "classification" else 1

        # TODO:
        self.abl: str = args.abl
        self.with_control = args.with_control

        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.len_dict = {
            "l": self.text_out,
            "v": self.video_hidden,
            "a": self.audio_hidden
        }
        print("len(self.abl) :",len(self.abl) )
        post_fusion_dim = (self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1)
        self.post_fusion_layer_1 = nn.Linear(post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
      # todo
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim + 1 if self.with_control else self.post_fusion_dim, self.output_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, text_x, audio_x, video_x, control_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)
        tensor_dict = {}

        audio_h = self.audio_subnet(audio_x)
        tensor_dict['a'] = audio_h

        video_h = self.video_subnet(video_x)
        tensor_dict['v'] = video_h

        text_h = self.text_subnet(text_x)
        tensor_dict['l'] = text_h
        # batch_size = audio_h.data.shape[0]
        batch_size = tensor_dict[self.abl[0]].shape[0]

        add_one = torch.ones(size=(batch_size, 1), requires_grad=False).type_as(tensor_dict[self.abl[0]]).to(tensor_dict[self.abl[0]].device)
        for k in tensor_dict.keys():
            tensor_dict[k] = torch.cat([add_one, tensor_dict[k]], dim=1)


        fusion_tensor = torch.bmm(tensor_dict[self.abl[0]].unsqueeze(2),  tensor_dict[self.abl[1]].unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (self.len_dict[self.abl[0]] + 1) * (self.len_dict[self.abl[1]] + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, tensor_dict[self.abl[-1]].unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        #todo
        if self.with_control:
            tensor_control = control_x.unsqueeze(1)
            post_fusion_y_2 = torch.cat((post_fusion_y_2, tensor_control), dim=1)
            output = self.post_fusion_layer_3(post_fusion_y_2)
        else:
            output = self.post_fusion_layer_3(post_fusion_y_2)


        if self.output_dim == 1: # regression
            output = torch.sigmoid(output)
            output = output * self.output_range + self.output_shift

        res = {
            'M': output
        }

        return res
