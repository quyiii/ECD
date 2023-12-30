import torch
import torch.nn as nn

class Decouple_Loss(nn.Module):
    def __init__(self, batch_nor, batch_abn, w_bag) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.batch_nor = batch_nor
        self.batch_abn = batch_abn
        self.w_bag = w_bag
        
    def forward(self, result, label):
        normal_label = label[:self.batch_nor]
        anomaly_label = label[self.batch_nor:]

        score_ee = result['bag_score_ee']
        score_es = result['bag_score_es']
        score_se = result['bag_score_se']
        score_ss = result['bag_score_ss']

        loss_ee = self.bce(score_ee, label)
        loss_ss = self.bce(
            score_ss,
            torch.cat((normal_label, normal_label)).to(score_ee.device)
        )

        loss_es = self.bce(
            score_es,
            torch.cat((normal_label, normal_label)).to(score_ee.device)
        )
        
        loss_se = self.bce(
            score_se,
            torch.cat((normal_label, normal_label)).to(score_ee.device)
        )

        loss_contrastive = 0
        for w, l in zip(self.w_bag, [loss_ee, loss_ss, loss_es, loss_se]):
            loss_contrastive += w * l

        decouple_loss = loss_contrastive

        return decouple_loss