import os
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

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
    def __init__(self, root_dir, source, target, source_mean_g = None, source_std_g = None, target_mean_g = None, target_std_g = None):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = []
        self.source_mcep = []
        self.source_time_frames = []
        self.target_mcep = []
        self.target_time_frames = []
        self.source_mean = source_mean_g
        self.source_std = source_std_g
        self.target_mean = target_mean_g
        self.target_std = target_std_g
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
            mcep = data['mcep']
            time_frames = data['time_frames']
            if speaker_id == source:
                self.source_mcep.append(mcep)
                self.source_time_frames.append(time_frames)
            elif speaker_id == target:
                self.target_mcep.append(mcep)
                self.target_time_frames.append(time_frames)
        self.normalize()

    def normalize(self):
        merged_source_mcep = np.concatenate(self.source_mcep, axis=0)
        if self.source_mean is None:
            self.source_mean = np.mean(merged_source_mcep)
            self.source_std = np.std(merged_source_mcep)
        for i in range(len(self.source_mcep)):
            self.source_mcep[i] = (self.source_mcep[i] - self.source_mean) / (self.source_std + 1e-8)
        merged_target_mcep = np.concatenate(self.target_mcep, axis=0)
        if self.target_mean is None:
            self.target_mean = np.mean(merged_target_mcep)
            self.target_std = np.std(merged_target_mcep)
        for i in range(len(self.target_mcep)):
            self.target_mcep[i] = (self.target_mcep[i] - self.target_mean) / (self.target_std + 1e-8)
    def getSourceNorm(self, source = True):
        if source:
            return self.source_mean, self.source_std
        else:
            return self.target_mean, self.target_std

    def normalizeMcep(self, mcep, source=True):
        if source:
            mcep = (mcep - self.source_mean) / self.source_std
        else:
            mcep = (mcep - self.target_mean) / self.target_std
        return mcep

    def denormalizeMcep(self, mcep, source=True):
        if source:
            mcep = mcep * self.source_std + self.source_mean
        else:
            mcep = mcep * self.target_std + self.target_mean
        return mcep

    def __len__(self):
        return len(self.source_mcep) + len(self.target_mcep)

def getId(label):
    return all_labels.index(label)

def getMcepSlice(mcep):
    if(len(mcep.shape) == 3):
        mcep = mcep.squeeze(0)
    max_start = max(0, mcep.shape[0] - 128)
    start = random.randint(0, max_start)
    return mcep[start:start + 128, :]

class PitchDataset(Dataset):
    def __init__(self, root_dir, source, target):
        self.root_dir = root_dir
        self.source_file_list = None
        self.target_file_list = None
        self.source_norm_log_f0 = []
        self.source_mean_log_f0 = []
        self.source_std_log_f0 = []
        self.source_time_frames = []
        self.source_log_f0_contours = []
        self.target_norm_log_f0 = []
        self.target_mean_log_f0 = []
        self.target_std_log_f0 = []
        self.target_time_frames = []
        self.target_log_f0_contours = []
        self.getData(source, target)

    def getFileListAndSpeakers(self, source, target):
        source_file_list = []
        target_file_list = []
        speaker_ids = set()
        for subdir, _, files in os.walk(self.root_dir):
            speaker_id = os.path.basename(subdir)
            if speaker_id == source:
                for file in files:
                    if file.endswith('.npz'):
                        file_path = os.path.join(subdir, file)
                        source_file_list.append(file_path)
                        speaker_ids.add(speaker_id)
            elif speaker_id == target:
                for file in files:
                    if file.endswith('.npz'):
                        file_path = os.path.join(subdir, file)
                        target_file_list.append(file_path)
                        speaker_ids.add(speaker_id)
        self.source_file_list = source_file_list
        self.target_file_list = target_file_list

    def getData(self, source, target):
        self.getFileListAndSpeakers(source, target)
        for idx, file in enumerate(self.source_file_list):
            data = np.load(file)
            norm_log_f0 = data['norm_log_f0']
            mean_log_f0 = data['mean_log_f0']
            std_log_f0 = data['std_log_f0']
            log_f0 = data['log_f0']
            time_frames = data['time_frames']
            self.source_norm_log_f0.append(norm_log_f0)
            self.source_mean_log_f0.append(mean_log_f0)
            self.source_std_log_f0.append(std_log_f0)
            self.source_log_f0_contours.append(log_f0)
            self.source_time_frames.append(time_frames)
        for idx, file in enumerate(self.target_file_list):
            data = np.load(file)
            norm_log_f0 = data['norm_log_f0']
            mean_log_f0 = data['mean_log_f0']
            std_log_f0 = data['std_log_f0']
            log_f0 = data['log_f0']
            time_frames = data['time_frames']
            self.target_norm_log_f0.append(norm_log_f0)
            self.target_mean_log_f0.append(mean_log_f0)
            self.target_std_log_f0.append(std_log_f0)
            self.target_log_f0_contours.append(log_f0)
            self.target_time_frames.append(time_frames)

    def logf0Statistics(self, log_f0s):
        log_f0s_concatenated = np.concatenate(log_f0s)
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()
        return log_f0s_mean, log_f0s_std

    def pitchConversion(self, log_f0):
        source_mean_log_f0, source_std_log_f0 = self.logf0Statistics(self.source_log_f0_contours)
        target_mean_log_f0, target_std_log_f0 = self.logf0Statistics(self.target_log_f0_contours)
        f0_converted = np.exp((log_f0 - source_mean_log_f0) / source_std_log_f0 * target_std_log_f0 + target_mean_log_f0)
        return f0_converted