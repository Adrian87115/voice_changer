import os
import scipy.io
from torch.utils.data import Dataset
import torch.nn.functional as f

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = None
        self.labels = None
        self.norm_log_f0 = None
        self.mcc = None
        self.source_parameter = None
        self.getData()
        # self.display_first_element()

    def getFileListAndSpeakers(self):
        file_list = []
        speaker_ids = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(subdir, file)
                    speaker_id = os.path.basename(os.path.dirname(file_path))
                    file_list.append(file_path)
                    speaker_ids.append(speaker_id)
        self.labels = speaker_ids
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