import torch.nn as nn
from .TFN import TFN



class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            'tfn': TFN,
        }
        lastModel = self.MODEL_MAP[args['model_name']]
        print(args)
        self.model_name = args['model_name']
        self.Model = lastModel(args)

        self.with_control = args.with_control


    def forward(self, text_x, audio_x, video_x,control_x= None ,meta_info = None ,  Y= None , *args,**kwargs):
        if self.with_control:
            return  self.Model(text_x, audio_x, video_x, control_x,*args, **kwargs)
        else:
            return self.Model(text_x, audio_x, video_x,  *args, **kwargs)
