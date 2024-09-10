import os
import scipy.io
from torch.utils.data import Dataset
import torch

class AudioDataset(Dataset):#shape of embedded torch.Size([1, 12, 36, 512]), wrong that is not adjustable for testing
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = []
        self.unique_labels = None
        self.speaker_emb = {}
        self.norm_log_f0 = None
        self.mcc = None
        self.source_parameter = None
        self.num_speakers = None
        self.getData()

    def getEmbbededLabels(self):
        self.embedded_labels = {}


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
        for file in self.file_list:
            mat_data = scipy.io.loadmat(file)
            norm_log_f0 = mat_data['norm_log_f0']
            mcc = mat_data['mcc']
            source_parameter = mat_data['source_parameter']
            norm_log_f0_list.append(norm_log_f0)
            mcc_list.append(mcc)
            source_parameter_list.append(source_parameter)
        self.norm_log_f0 = norm_log_f0_list
        self.mcc = mcc_list
        self.source_parameter = source_parameter_list
        self.labelEmb()

    def labelEmb(self):
        for i in range(self.num_speakers):
            batch_size = 1
            speaker_index = torch.tensor([i], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            speaker_embedding = torch.zeros(batch_size, self.num_speakers, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).scatter_(1, speaker_index.unsqueeze(1), 1)
            speaker_embedding = speaker_embedding.unsqueeze(2).unsqueeze(3)
            speaker_embedding = speaker_embedding.expand(speaker_embedding.size(0), speaker_embedding.size(1), 36, 512)
            self.speaker_emb[self.labels[i]] = speaker_embedding

    def __getitem__(self, idx):
        mcc = torch.tensor(self.mcc[idx], dtype=torch.float32)
        speaker_id = self.unique_labels.index(self.labels[idx])
        return mcc, torch.tensor(speaker_id, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)