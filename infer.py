import torch
import utils
import prettytable
import numpy as np
import os

from models import WSAD
import torch.utils.data as data
from collections import defaultdict

from datasets import get_dataset
from options import parse_args
from sklearn.metrics import roc_curve,auc,precision_recall_curve

from test import get_test_bz, get_frame_gt

def get_anomaly_mask(datasetname):
    if 'ucf' in datasetname:
        anomaly_mask = np.load('frame_label/ucf_anomaly_mask.npy')
    elif 'xd' in datasetname:
        anomaly_mask = np.load('frame_label/xd_anomaly_mask.npy')
    else:
        raise RuntimeError(f'Unknown Dataset: {datasetname}')
    return anomaly_mask

def get_predicts(test_loader, net, datasetname):
    load_iter = iter(test_loader)
    test_bz = get_test_bz(datasetname)

    frame_predict = defaultdict(lambda: [])
    
    for i in range(len(test_loader.dataset)//test_bz):
        _data, _label = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()

        if 'ucf' in datasetname and len(_data[0]) > 6000:
            len_step = len(_data[0])
            target_len = 2000
            _data = _data[:, torch.linspace(0, len_step-1, target_len, dtype=torch.int32).cuda()]

            res = net(_data)

            for k, v in res.items():
                res[k] = v[:, torch.linspace(0, target_len-1, len_step, dtype=torch.int32).cuda()]

                # res[k] = res[k] - v.min(1, keepdim=True)
        else:
            res = net(_data)   
        
        for head_name, head_predict in res.items(): 
            a_predict = head_predict.cpu().numpy().mean(0)
            fpre_ = np.repeat(a_predict, 16)         
            frame_predict[head_name].append(fpre_)

    for head_name, preds in frame_predict.items():
        frame_predict[head_name] = np.concatenate(preds, axis=0)

    return frame_predict

def get_sub_metrics(frame_predict, frame_gt, datasetname):
    anomaly_mask = get_anomaly_mask(datasetname)
    sub_predict = frame_predict[anomaly_mask]
    sub_gt = frame_gt[anomaly_mask]
    
    fpr,tpr,_ = roc_curve(sub_gt, sub_predict)
    auc_sub = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(sub_gt, sub_predict)
    ap_sub = auc(recall, precision)
    return auc_sub, ap_sub

def get_metrics(frame_predicts, frame_gt, datasetname):
    metrics = {}
    for head_name, frame_predict in frame_predicts.items():
        metrics[head_name] = {}
        
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        metrics[head_name]['AUC'] = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        metrics[head_name]['AP'] = auc(recall, precision)

        auc_sub, ap_sub = get_sub_metrics(frame_predict, frame_gt, datasetname)
        metrics[head_name]['AUC_sub'] = auc_sub
        metrics[head_name]['AP_sub'] = ap_sub
    return metrics

def test(net, test_loader, datasetname, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        frame_gt = get_frame_gt(datasetname)
        frame_predicts = get_predicts(test_loader, net, datasetname)
        metrics = get_metrics(frame_predicts, frame_gt, datasetname)

        for head_name, head_metrics in metrics.items():
            for score_name, score in head_metrics.items():
                new_score = score * 100
                metrics[head_name][score_name] = new_score

        return metrics

model_paths = []

if __name__ == '__main__':
    args = parse_args()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(in_channel=args.in_channel, flag='Test', args=args)
    net = net.cuda()
    dataset = get_dataset(args.dataset)

    test_loader = data.DataLoader(
        dataset(root_dir=args.root_dir, mode='Test', num_segments=args.num_segments, len_feature=args.in_channel, is_normal=False),
        batch_size=get_test_bz(args.dataset),
        shuffle=False, num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    if len(model_paths) == 0:
        model_paths.append(args.model_path)
    
    for model_path in model_paths:
        res = test(net, test_loader, args.dataset, model_path)
        pt = prettytable.PrettyTable()
        pt.field_names = ['Head', 'AUC', 'AP', 'AUC_sub', 'AP_sub']
        for head_name, head_scores in res.items():
            pt.add_row([head_name, head_scores['AUC'], head_scores['AP'], head_scores['AUC_sub'], head_scores['AP_sub']])
        print(model_path)
        print(pt)
