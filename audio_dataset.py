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
        self.original_mcc_size = None
        self.num_speakers = None
        self.getData()

    def getFileListAndSpeakers(self):
        file_list = []
        speaker_ids = set()
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mat'):
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
        original_mcc_size_list = []
        for file in self.file_list:
            mat_data = scipy.io.loadmat(file)
            norm_log_f0 = mat_data['norm_log_f0']
            mcc = mat_data['mcc']
            source_parameter = mat_data['source_parameter']
            original_mcc_size = mat_data['original_mcc_size']
            norm_log_f0_list.append(norm_log_f0)
            mcc_list.append(mcc)
            source_parameter_list.append(source_parameter)
            original_mcc_size_list.append(original_mcc_size)
        self.norm_log_f0 = norm_log_f0_list
        self.mcc = mcc_list
        self.source_parameter = source_parameter_list
        self.original_mcc_size = original_mcc_size_list
        self.labelEmb()

    def labelEmb(self):
        global all_labels
        for label in self.labels:
            if label in all_labels:
                label_index = all_labels.index(label)  # Get the index from the global list
                speaker_embedding = torch.zeros(1, 1, 36, 512,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                row = label_index % 36
                col = label_index % 512
                speaker_embedding[0, 0, row, col] = 1
                self.speaker_emb[label] = speaker_embedding

    def __getitem__(self, idx):
        mcc = torch.tensor(self.mcc[idx], dtype=torch.float32)
        speaker_id = all_labels.index(self.labels[idx])
        return mcc, torch.tensor(speaker_id, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)

    def getMeanNormLogf0ForSpeaker(self, speaker_label):
        norm_log_f0_values = []

        # Iterate over labels to collect F0 values for the specified speaker
        for idx, label in enumerate(self.labels):
            if label == speaker_label:
                # Convert to NumPy array and filter out NaNs
                f0 = np.array(self.norm_log_f0[idx])
                valid_f0 = f0[~np.isnan(f0)]  # Remove NaNs
                f0_tensor = torch.tensor(valid_f0)  # Convert to PyTorch tensor
                resized_f0 = resizef0ToFixedLength(f0_tensor, target_length=512)  # Resize to 512
                norm_log_f0_values.append(resized_f0)

        # Visualize the first valid f0 sample (after resizing)
        norm_log_f0_0 = np.array(self.norm_log_f0[0])  # Convert to NumPy array
        valid_indices = ~np.isnan(norm_log_f0_0)
        valid_data = norm_log_f0_0[valid_indices]

        # plt.figure(figsize=(10, 5))
        # plt.plot(valid_data)
        # plt.title("Normalized Log F0 (Valid Data) for the First Sample")
        # plt.grid(True)
        # plt.show()

        # Stack non-NaN F0 values and calculate the mean
        if norm_log_f0_values:  # Check if any values were added
            norm_log_f0_values = torch.stack(norm_log_f0_values)
            mean_norm_log_f0 = torch.mean(norm_log_f0_values, dim=0)
        else:
            mean_norm_log_f0 = torch.tensor([])  # Handle empty case

        # Visualize the mean normalized log F0 after resizing
        # plt.figure(figsize=(10, 5))
        # plt.plot(mean_norm_log_f0)
        # plt.title("Mean Normalized Log F0 for Speaker After Resizing to 512")
        # plt.grid(True)
        # plt.show()

        return mean_norm_log_f0

    def getSpeakerEmbedding(self, label):
        return self.speaker_emb[label]


def resizef0ToFixedLength(tensor, target_length=512):
    current_length = tensor.size(0)

    if current_length > target_length:
        return tensor[:target_length]  # Truncate to the target length
    elif current_length < target_length:
        padding = torch.full((target_length - current_length,), tensor[-1])  # Pad with the last value
        return torch.cat((tensor, padding))
    else:
        return tensor  # No change if already the correct length


def getSpeakerEmbeddingFromLabel(label):
    global all_labels

    # Check if the label exists in the global list
    if label not in all_labels:
        raise ValueError(f"Label '{label}' not found in the list of speakers.")

    # Get the index of the speaker in the global list
    label_index = all_labels.index(label)

    # Create an embedding tensor of size (1, 1, 36, 512)
    embedding = torch.zeros(1, 1, 36, 512, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Calculate the row and column position based on the index
    row = label_index % 36
    col = label_index % 512

    # Set the corresponding position to 1
    embedding[0, 0, row, col] = 1

    return embedding