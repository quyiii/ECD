import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from .enhancer import Enhancer
from .clsheads import get_clsHead
from .memory import Memory_Unit
from .attention import ChannelOnlyAttention, TemporalOnlyAttention

class WSAD(Module):
    def __init__(self, in_channel, flag, args):
        super().__init__()
        self.in_channel = in_channel
        self.flag = flag
        self.args = args
        
        self.dropout = args.dropout
        self.batch_nor = args.batch_nor
        self.batch_abn = args.batch_abn
        self.model_type = args.model_type

        self.build_layers()

    def build_layers(self):
        self.enhancer = Enhancer(self.in_channel, self.dropout)
        self.in_channel = 512 if self.model_type == 'rtfm' else 1024
        self.clshead = get_clsHead(self.model_type)(self.in_channel)
        if self.model_type == 'ur':
            self.triplet = nn.TripletMarginLoss(margin=1)
            self.distance_triplet = nn.TripletMarginLoss(margin=15)
            self.Amemory = Memory_Unit(nums=60, dim=512)
            self.Nmemory = Memory_Unit(nums=60, dim=512)
            self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
            self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        
        self.channel_attener = ChannelOnlyAttention(self.in_channel)
        if self.model_type == 'rtfm':
            self.temporal_attener = TemporalOnlyAttention(self.in_channel)

    def get_score(self, x, ncrops=None, atten=None):
        score = self.clshead(x)

        if atten is not None:
            score = score * atten

        if ncrops is not None:
            b = score.shape[0] // ncrops
            score = score.view(b, ncrops, -1).mean(1)
        else:
            score = score.squeeze(-1)
        return score

    def get_select_reference(self, score, feat):
        # feat: B x T x C
        reference = torch.norm(feat, p=2, dim=-1) * score
        return reference
    
    def get_select_idx(self, reference, topk=None):
        if topk is None:
            t = reference.shape[-1]
            topk = t // 16 + 1
        
        select_idx = torch.topk(reference, k=topk, dim=-1)[1]
        return select_idx

    def get_select_val(self, val, reference=None, select_idx=None, topk=None):
        if reference is None:
            reference = val
        
        if select_idx is None:
            select_idx = self.get_select_idx(reference, topk)

        select_val = torch.gather(val, dim=-1, index=select_idx)
        
        select_ref = torch.gather(reference, dim=-1, index=select_idx)
        return select_val, select_ref

    def _decouple(self, x, b, n, tatten_e=None):
        catten_e = self.channel_attener(x)
        if tatten_e is None:
            tatten_e = self.temporal_attener(x)

        catten_s = 1 - catten_e
        tatten_s = 1 - tatten_e

        feat_e = (x * catten_e).permute(0, 2, 1)
        feat_s = (x * catten_s).permute(0, 2, 1)

        score_e = self.get_score(feat_e, n)
        score_s = self.get_score(feat_s, n)

        bag_score_ee = (score_e * F.softmax(tatten_e.view(b,n,-1).mean(1), dim=-1)).sum(dim=-1)
        bag_score_es = (score_e * F.softmax(tatten_s.view(b,n,-1).mean(1), dim=-1)).sum(dim=-1)
        bag_score_se = (score_s * F.softmax(tatten_e.view(b,n,-1).mean(1), dim=-1)).sum(dim=-1)
        bag_score_ss = (score_s * F.softmax(tatten_s.view(b,n,-1).mean(1), dim=-1)).sum(dim=-1)

        bag_scores = {
            'bag_score_ee': bag_score_ee,
            'bag_score_es': bag_score_es,
            'bag_score_se': bag_score_se,
            'bag_score_ss': bag_score_ss,
        }
        
        score_e = score_e * tatten_e.view(b,n,-1).mean(1)
        return feat_e, score_e, bag_scores

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def norm(self, data):
        l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
        return torch.div(data, l2)

    def forward_ur(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        
        x = self.enhancer(x)

        if self.flag == 'Train':
            N_x = x[:self.batch_nor*n]                  #### Normal part
            A_x = x[self.batch_nor*n:]                  #### Abnormal part
            A_att, A_aug = self.Amemory(A_x)   ###bt,btd,   anomaly video --->>>>> Anomaly memeory  at least 1 [1,0,0,...,1]
            N_Aatt, N_Aaug = self.Nmemory(A_x) ###bt,btd,   anomaly video --->>>>> Normal memeory   at least 0 [0,1,1,...,1]

            A_Natt, A_Naug = self.Amemory(N_x) ###bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
            N_att, N_aug = self.Nmemory(N_x)   ###bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]
    
            _, A_index = torch.topk(A_att, t//16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(self.batch_abn,n,-1).mean(1)
            
            _, N_index = torch.topk(N_att, t//16 + 1, dim=-1)
            anchor_nx=torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(self.batch_nor,n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, t//16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(self.batch_abn,n,-1).mean(1)

            triplet_margin_loss = self.triplet(self.norm(anchor_nx), self.norm(positivte_nx), self.norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).reshape(self.batch_nor,n,t//16+1,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).reshape(self.batch_abn,n,t//16+1,-1).mean(1)
            
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            nor_mag = torch.norm(anchor_nx_new, p=2, dim=-1).view(-1)
            abn_mag = torch.norm(negative_ax_new, p=2, dim=-1).view(-1)
            anchor_mag = torch.zeros_like(nor_mag)
            
            distance = self.distance_triplet(anchor_mag, nor_mag, abn_mag)
            
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)

            feat_e, score_e, bag_scores = self._decouple(x.permute(0,2,1), b, n, torch.cat((A_Natt, A_att), dim=0))
            pre_att = score_e

            res = {
                'frame': pre_att,
                'triplet_margin': triplet_margin_loss,
                'kl_loss': kl_loss, 
                'distance': distance,
                'A_att': A_att.reshape((self.batch_abn, n, -1)).mean(1),
                "N_att": N_att.reshape((self.batch_nor, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((self.batch_nor, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((self.batch_abn, n, -1)).mean(1),
                'bag_scores': bag_scores
            }

            return res
        else:
            A_att, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)


            feat_e, score_e, bag_scores = self._decouple(x.permute(0,2,1), b, n, A_att)
            pre_att = score_e

            return {
                'frame': pre_att,
            }
    
    def forward_rtfm(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        
        x = self.enhancer(x)

        
        feat_e, score_e, bag_scores = self._decouple(x.permute(0, 2, 1), b, n)

        feat_e = feat_e.view(b, n, t, -1).mean(1)

        reference = self.get_select_reference(score_e, feat_e)

        if self.flag == "Train":
            # BN x T x C
            nor_score = score_e[:self.batch_nor]
            abn_score = score_e[self.batch_nor:]

            nor_reference = reference[:self.batch_nor]
            abn_reference = reference[self.batch_nor:]

            nor_select_idx = self.get_select_idx(nor_reference)
            abn_select_idx = self.get_select_idx(abn_reference)

            select_nor_score, select_nor_ref = self.get_select_val(nor_score, nor_reference, nor_select_idx)
            select_abn_score, select_abn_ref = self.get_select_val(abn_score, abn_reference, abn_select_idx)

            res = {
                'mil': {
                    'nor_score': select_nor_score,
                    'abn_score': select_abn_score,
                },
                'triplet': {
                    'nor_ref': select_nor_ref,
                    'abn_ref': select_abn_ref,
                },
                'bag_scores': bag_scores
            }

            return res
        else:
            return {
                'frame': score_e
            }

    def forward(self, x):
        if self.model_type == 'ur':
            results = self.forward_ur(x)
        elif self.model_type == 'rtfm':
            results = self.forward_rtfm(x)
        else:
            raise RuntimeError(f'Unknown model type: {self.model_type}')
        return results
