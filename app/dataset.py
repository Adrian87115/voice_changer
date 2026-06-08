import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Class to obtain the MCEP data, calculate the statistics, perform normalization and denormalization
class AudioDataset:
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
        merged_source_mcep = np.concatenate(self.source_mcep, axis = 0)

        if self.source_mean is None:
            self.source_mean = np.mean(merged_source_mcep, axis = 0, keepdims = True)
            self.source_std = np.std(merged_source_mcep, axis = 0, keepdims = True)

        for i in range(len(self.source_mcep)):
            self.source_mcep[i] = (self.source_mcep[i] - self.source_mean) / (self.source_std + 1e-8)

        merged_target_mcep = np.concatenate(self.target_mcep, axis = 0)

        if self.target_mean is None:
            self.target_mean = np.mean(merged_target_mcep, axis = 0, keepdims = True)
            self.target_std = np.std(merged_target_mcep, axis = 0, keepdims = True)

        for i in range(len(self.target_mcep)):
            self.target_mcep[i] = (self.target_mcep[i] - self.target_mean) / (self.target_std + 1e-8)

    def getSourceNorm(self, source = True):
        if source:
            return self.source_mean, self.source_std
        else:
            return self.target_mean, self.target_std

    def normalizeMcep(self, mcep, source = True):
        if source:
            mcep = (mcep - self.source_mean) / (self.source_std + 1e-8)
        else:
            mcep = (mcep - self.target_mean) / (self.target_std + 1e-8)

        return mcep

    def denormalizeMcep(self, mcep, source = True):
        if source:
            mcep = mcep * self.source_std + self.source_mean
        else:
            mcep = mcep * self.target_std + self.target_mean

        return mcep

    def __len__(self):
        return len(self.source_mcep) + len(self.target_mcep)

# Class to obtain the pitch data, calculate the statistics, and perform pitch conversion
class PitchDataset:
    def __init__(self, root_dir, source, target):
        self.root_dir = root_dir
        self.source_file_list = None
        self.target_file_list = None
        self.source_f0 = []
        self.source_time_frames = []
        self.target_f0 = []
        self.target_time_frames = []
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

        for file in self.source_file_list:
            data = np.load(file)
            f0 = data['f0']
            time_frames = data['time_frames']
            self.source_f0.append(f0)
            self.source_time_frames.append(time_frames)
            
        for file in self.target_file_list:
            data = np.load(file)
            f0 = data['f0']
            time_frames = data['time_frames']
            self.target_f0.append(f0)
            self.target_time_frames.append(time_frames)

    def logf0Statistics(self, f0s):
        f0s_concatenated = np.concatenate(f0s)
        voiced_f0s = f0s_concatenated[f0s_concatenated > 0]
        log_f0s = np.log(voiced_f0s)
        log_f0s_mean = log_f0s.mean()
        log_f0s_std = log_f0s.std()
        return log_f0s_mean, log_f0s_std

    def pitchConversion(self, log_f0):
        source_mean_log_f0, source_std_log_f0 = self.logf0Statistics(self.source_f0)
        target_mean_log_f0, target_std_log_f0 = self.logf0Statistics(self.target_f0)
        f0_converted = np.exp((log_f0 - source_mean_log_f0) / (source_std_log_f0 + 1e-8) * target_std_log_f0 + target_mean_log_f0)
        return f0_converted
    
# Main dataset to prepare data for the DataLoader
class MCEPDataset(Dataset):
    def __init__(self, mcep_data, label_id):
        self.mcep_data = [torch.tensor(mcep, dtype = torch.float32) for mcep in mcep_data]
        self.label_id = label_id

    def __len__(self):
        return len(self.mcep_data)

    def __getitem__(self, idx):
        return self.mcep_data[idx], self.label_id