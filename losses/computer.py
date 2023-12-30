from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ur_loss import UR_Loss
from .rtfm_loss import RTFM_Loss
from .decouple_loss import Decouple_Loss

class LossComputer(nn.Module):
    def __init__(self, args):
        super(LossComputer, self).__init__()
        self.args = args
        self.model_type = args.model_type
        self.batch_nor = args.batch_nor
        self.batch_abn = args.batch_abn
        self.w_bag = args.w_bag

        self.loss = self.build_loss()
        
        self.decouple_loss = Decouple_Loss(self.batch_nor, self.batch_abn, self.w_bag)

    def build_loss(self):
        if self.model_type == 'ur':
            return UR_Loss()
        elif self.model_type == 'rtfm':
            return RTFM_Loss()
        else:
            raise RuntimeError(f'Unknown model type: {self.model_type}')
        
    def forward(self, result, label, step):
        loss, losses = self.loss(result, label, step)

        if 'bag_scores' in result:
            bag_res = result['bag_scores']
            decouple_loss = self.decouple_loss(bag_res, label)
            loss += decouple_loss
            losses['decouple_loss'] = decouple_loss

        losses['total_loss'] = loss
        return loss, losses