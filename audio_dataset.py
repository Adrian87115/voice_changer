import os
from torch.utils.data import Dataset
import numpy as np

all_labels = ["VCC2SF1",
              "VCC2SF2",
              "VCC2SF3",
              "VCC2SF4",
              "VCC2SM1",
              "VCC2SM2",
              "VCC2SM3",
              "VCC2SM4",
              "VCC2TF1",
              "VCC2TF2",
              "VCC2TM1",
              "VCC2TM2"]

target_labels = ["VCC2TF1",
                 "VCC2TF2",
                 "VCC2TM1",
                 "VCC2TM2"]

class AudioDataset(Dataset):
    def __init__(self, root_dir, source, target):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = []
        self.source_norm_log_f0 = []
        self.source_mcc = []
        self.source_parameter = []
        self.source_time_frames = []
        self.target_norm_log_f0 = []
        self.target_mcc = []
        self.target_parameter = []
        self.target_time_frames = []
        self.getData(source, target)

    def getFileListAndSpeakers(self, source, target):
        file_list = []
        speaker_ids = set()
        for subdir, _, files in os.walk(self.root_dir):
            speaker_id = os.path.basename(subdir)
            if speaker_id in [source, target]:
                for file in files:
                    if file.endswith('.npz'):
                        file_path = os.path.join(subdir, file)
                        file_list.append(file_path)
                        speaker_ids.add(speaker_id)
                        self.labels.append(speaker_id)
        self.file_list = file_list

    def getData(self, source, target):
        self.getFileListAndSpeakers(source, target)
        for idx, file in enumerate(self.file_list):
            speaker_id = self.labels[idx]
            data = np.load(file)
            norm_log_f0 = data['norm_log_f0']
            mcc = data['mcc']
            source_parameter = data['source_parameter']
            time_frames = data['time_frames']
            if speaker_id == source:
                self.source_norm_log_f0.append(norm_log_f0)
                self.source_mcc.append(mcc)
                self.source_parameter.append(source_parameter)
                self.source_time_frames.append(time_frames)
            elif speaker_id == target:
                self.target_norm_log_f0.append(norm_log_f0)
                self.target_mcc.append(mcc)
                self.target_parameter.append(source_parameter)
                self.target_time_frames.append(time_frames)

    def __len__(self):
        return len(self.source_mcc) + len(self.target_mcc)

def getId(label):
    return all_labels.index(label)