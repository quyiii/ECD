import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode=mode
        
        self.num_segments = num_segments
        self.len_feature = len_feature
        if type(root_dir) == list:
            self.feature_path, self.audio_path = root_dir
        else:
            self.audio_path = None
            self.feature_path = root_dir
        
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_audio_name(self, vid_name):
        file_dir, file_name = vid_name.split('/')
        audio_name = "{video_id}__vggish.npy".format(video_id=file_name[:-7])
        return "{file_dir}/{audio_name}".format(file_dir=file_dir, audio_name=audio_name)

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        audio_name = self.get_audio_name(vid_name) if self.audio_path is not None else None
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path, vid_name)).astype(np.float32)
        audio_feature = np.load(os.path.join(self.audio_path, audio_name)).astype(np.float32) if self.audio_path is not None else None
        if self.mode == "Train":
            video_feature = utils.process_feature(video_feature, self.num_segments, 'xd')
            audio_feature = utils.process_feature(audio_feature, self.num_segments, 'xd') if self.audio_path is not None else None
        if audio_feature is not None:
            video_feature = np.concatenate([video_feature, audio_feature], axis=1)
        return video_feature, label    
