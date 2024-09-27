import os
import scipy.io
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
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
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = []
        self.unique_labels = None
        self.one_hot_labels = {}
        self.norm_log_f0 = None
        self.mcc = None
        self.source_parameter = None
        self.time_frames_list = None
        self.num_speakers = None
        self.getData()

    def getFileListAndSpeakers(self):
        file_list = []
        speaker_ids = set()
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(subdir, file)
                    speaker_id = os.path.basename(os.path.dirname(file_path))
                    file_list.append(file_path)
                    speaker_ids.add(speaker_id)
                    self.labels.append(speaker_id)
        self.unique_labels = sorted(list(speaker_ids))
        self.num_speakers = len(self.unique_labels)
        self.file_list = file_list

    def getData(self):
        self.getFileListAndSpeakers()
        norm_log_f0_list = []
        mcc_list = []
        source_parameter_list = []
        time_frames_list = []
        for file in self.file_list:
            data = np.load(file)
            norm_log_f0 = data['norm_log_f0']
            mcc = data['mcc']
            source_parameter = data['source_parameter']
            tf = data['time_frames']
            norm_log_f0_list.append(norm_log_f0)
            mcc_list.append(mcc)
            source_parameter_list.append(source_parameter)
            time_frames_list.append(tf)
        self.norm_log_f0 = norm_log_f0_list
        self.mcc = mcc_list
        self.source_parameter = source_parameter_list
        self.time_frames_list = time_frames_list
        self.generateOneHotLabels()

    def generateOneHotLabels(self):
        num_labels = self.num_speakers
        for idx, label in enumerate(self.unique_labels):
            one_hot = torch.zeros(num_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            one_hot[idx] = 1.0  # Set the one-hot position for the current label
            self.one_hot_labels[label] = one_hot


    def __getitem__(self, idx):
        mcc = torch.tensor(self.mcc[idx], dtype=torch.float32)
        label = self.labels[idx]
        if label in self.one_hot_labels:
            one_hot_label = self.one_hot_labels[label]
        else:
            one_hot_label = None
        return mcc, one_hot_label

    def __len__(self):
        return len(self.file_list)

    def getSpeakerOneHot(self, label):
        return self.one_hot_labels[label]

def getSpeakerOneHotFromLabel(label):
    global all_labels
    num_labels = len(all_labels)
    one_hot = torch.zeros(num_labels, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    label_index = all_labels.index(label)
    one_hot[label_index] = 1.0
    return one_hot
