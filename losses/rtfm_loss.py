import torch
import torch.nn as nn

class RTFM_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.triplet = nn.TripletMarginLoss(15, p=2)
    
    def forward(self, result, label, step):
        loss = 0
        losses = {}
        # mil
        if 'mil' in result:
            mil_res = result['mil']
            nor_score = mil_res['nor_score'].mean(-1)
            abn_score = mil_res['abn_score'].mean(-1)
            score_mil_loss = self.bce(torch.cat((nor_score, abn_score), dim=0), label)
            loss += score_mil_loss
            losses['mil_loss'] = score_mil_loss
        
        # triplet
        if 'triplet' in result:
            trip_res = result['triplet']
            nor_ref = trip_res['nor_ref']
            abn_ref = trip_res['abn_ref']
            b, t = nor_ref.shape
            anchor = torch.tensor([0])[..., None].expand(b*t, 1).cuda()
            nor_ref = nor_ref.view(b*t, 1)
            abn_ref = abn_ref.view(b*t, 1)
            
            score_triplet_loss = self.triplet(anchor, nor_ref, abn_ref)

            loss += score_triplet_loss
            losses['triplet_loss'] = score_triplet_loss

        return loss, losses
