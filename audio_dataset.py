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

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = []
        self.unique_labels = None
        self.speaker_emb = {}
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
        self.labelEmb()

    def labelEmb(self):
        global all_labels
        for label in self.labels:
            if label in all_labels:
                label_index = all_labels.index(label)  # Get the index from the global list
                speaker_embedding = torch.zeros(1, 1, 512, 36,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                row = label_index % 512
                col = label_index % 36
                speaker_embedding[0, 0, row, col] = 1
                self.speaker_emb[label] = speaker_embedding

    def __getitem__(self, idx):
        mcc = torch.tensor(self.mcc[idx], dtype=torch.float32)
        speaker_id = all_labels.index(self.labels[idx])
        return mcc, torch.tensor(speaker_id, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)

    def getSpeakerEmbedding(self, label):
        return self.speaker_emb[label]

def getSpeakerEmbeddingFromLabel(label):
    global all_labels
    if label not in all_labels:
        raise ValueError(f"Label '{label}' not found in the list of speakers.")
    label_index = all_labels.index(label)
    embedding = torch.zeros(1, 1, 512, 36, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    row = label_index % 512
    col = label_index % 36
    embedding[0, 0, row, col] = 1
    return embedding