import torch
import numpy as np

from sklearn.metrics import roc_curve,auc,precision_recall_curve

from collections import defaultdict

def get_frame_gt(datasetname):
    if 'ucf' in datasetname:
        frame_gt = np.load("frame_label/ucf_gt.npy")
    elif 'xd' in datasetname:
        frame_gt = np.load("frame_label/xd_gt.npy")
    else:
        raise RuntimeError(f'Unknown Dataset: {datasetname}')
    return frame_gt

def get_test_bz(datasetname):
    if 'ucf' in datasetname:
        test_bz = 10
    elif 'xd' in datasetname:
        test_bz = 5
    else:
        raise RuntimeError(f'Unknown Dataset: {datasetname}')
    return test_bz

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

def get_metrics(frame_predicts, frame_gt):
    metrics = {}
    for head_name, frame_predict in frame_predicts.items():
        metrics[head_name] = {}
        
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        metrics[head_name]['AUC'] = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        metrics[head_name]['AP'] = auc(recall, precision)
    return metrics

def test(net, test_loader, datasetname, model_file = None, out_name_formatter='{score_name}_{head_name}'):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        frame_gt = get_frame_gt(datasetname)
        frame_predicts = get_predicts(test_loader, net, datasetname)
        metrics = get_metrics(frame_predicts, frame_gt)

        scores = {}
        for head_name, head_metrics in metrics.items():
            for score_name, score in head_metrics.items():
                new_score = score * 100
                metrics[head_name][score_name] = new_score

                out_name = out_name_formatter.format(score_name=score_name, head_name=head_name)
                
                scores[out_name] = new_score

        return metrics, scores
