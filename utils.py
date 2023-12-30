import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype = np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def save_settings(file_root, args, net):
    setting_path = os.path.join(file_root, 'setting.log')
    with open(setting_path, 'w') as f:
        f.write(f'Arguments: \n{args}\n')
        f.write(f'Model Structure: \n{net}')

def save_best(test_info, best_scores, net, dataset_name, step, metric_name, file_root, model_root=None, out_name_formatter='{score_name}_{head_name}'):
    for head_name, head_scores in test_info.items():
        score_name = metric_name
        score = head_scores[score_name]

        score_file_root = os.path.join(file_root, head_name, score_name)
        if not os.path.exists(score_file_root):
            os.makedirs(score_file_root)

        model_file_root = os.path.join(model_root, head_name, score_name) if model_root is not None else score_file_root
        if not os.path.exists(model_file_root):
            os.makedirs(model_file_root)

        best_name = 'best_' + out_name_formatter.format(score_name=score_name, head_name=head_name)
        if score > best_scores[best_name]:
            file_path = os.path.join(score_file_root, '{}_best_record.txt'.format(dataset_name))
        
            save_best_record(file_path, step, head_scores)

            torch.save(net.state_dict(), os.path.join(model_file_root, "{}_best.pkl".format(dataset_name)))
        

def save_best_record(file_path, step, metric):
    with open(file_path, "w") as fo:
        fo.write("Step: {}\n".format(step))
        fo.write("AUC: {:.4f}\n".format(metric["AUC"]))
        fo.write("AP: {:.4f}\n".format(metric["AP"]))

def get_ucf_n_segments_mean(feature, num_segments):
    new_feat = np.zeros((num_segments, feature.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feature), num_segments + 1, dtype = np.int64)
    for i in range(num_segments):
        if r[i] != r[i+1]:
            new_feat[i,:] = np.mean(feature[r[i]:r[i+1],:], 0)
        else:
            new_feat[i:i+1,:] = feature[r[i]:r[i]+1,:]
    return new_feat

def get_xd_n_segments_mean(feature, num_segments):
    new_feature = np.zeros((num_segments, feature.shape[1])).astype(np.float32)
    sample_index = random_perturb(feature.shape[0], num_segments)
    for i in range(len(sample_index)-1):
        if sample_index[i] == sample_index[i+1]:
            new_feature[i,:] = feature[sample_index[i],:]
        else:
            new_feature[i,:] = feature[sample_index[i]:sample_index[i+1],:].mean(0)
            
    return new_feature

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def process_feature(feature, num_segments, datasetname):
    new_feature = None
    if datasetname == 'ucf':
        new_feature = get_ucf_n_segments_mean(feature, num_segments)
    elif datasetname == 'xd':
        new_feature = get_xd_n_segments_mean(feature, num_segments)
    else:
        RuntimeError('unknown dataset name')
    
    
    assert new_feature is not None
    return new_feature

def get_processed_path(video_path, segment_num, method_name):
    video_dir, video_name = video_path.split('/')
    video_dir = '{}_{}_{}'.format(video_dir, segment_num, method_name)
    return '{}/{}'.format(video_dir, video_name)