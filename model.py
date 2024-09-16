import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import scipy.io
from scipy.fftpack import idct
from scipy.interpolate import interp1d
import soundfile as sf
import generator as g
import discriminator as d
import domain_classifier as dc
import audio_dataset as ad
import utility as u
import warnings
warnings.filterwarnings("ignore")

class Model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_source_speakers = 8
        self.num_target_speakers = 4
        self.getData()
        self.generator = g.Generator(self.train_dataset.num_speakers).to(self.device)# maybe the problem is with giving id which will be used only once, maybe instead it should have only 4 from target, or not following the feeding architecture(gen to disc etc), or not using attributes in architecture of networks
        self.discriminator = d.Discriminator().to(self.device)
        self.domain_classifier = dc.DomainClassifier(self.num_target_speakers).to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.criterion_adv = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.g_scheduler = StepLR(self.g_optimizer, step_size=1, gamma=0.01)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=1, gamma=0.01)
        self.c_scheduler = StepLR(self.c_optimizer, step_size=1, gamma=0.1)
        self.top_score = float('inf')

    def saveModel(self):
        file_path = "saved_model.pth"
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'domain_classifier_state_dict': self.domain_classifier.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'c_optimizer_state_dict': self.c_optimizer.state_dict(),
            'top_score': self.top_score}, file_path)

    def loadModel(self):
        file_path = "saved_model.pth"
        checkpoint = torch.load(file_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
        self.top_score = checkpoint['top_score']

    def getData(self):
        path_train = "training_data/resized_audio" #first 8 are source, next 4 are target
        path_eval = "evaluation_data/resized_audio"
        path_ref = "reference_data/resized_audio" #used to compare original to created by listening to them
        self.train_dataset = ad.AudioDataset(path_train)
        self.eval_dataset = ad.AudioDataset(path_eval)
        self.ref_dataset = ad.AudioDataset(path_ref)

        train_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.train_dataset.mcc])
        eval_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.eval_dataset.mcc])
        ref_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.ref_dataset.mcc])

        def labelsToTensor(labels):
            if isinstance(labels[0], str):
                label_set = sorted(set(labels))
                label_to_index = {label: idx for idx, label in enumerate(label_set)}
                return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
            else:
                return torch.tensor(labels, dtype=torch.long)
        train_labels = labelsToTensor(self.train_dataset.labels)
        eval_labels = labelsToTensor(self.eval_dataset.labels)
        ref_labels = labelsToTensor(self.ref_dataset.labels)

        train_dataset = TensorDataset(train_mcc, train_labels)
        eval_dataset = TensorDataset(eval_mcc, eval_labels)
        ref_dataset = TensorDataset(ref_mcc, ref_labels)

        source_indices = torch.where(train_labels < self.num_source_speakers)[0]
        target_indices = torch.where((train_labels >= self.num_target_speakers) & (train_labels < self.train_dataset.num_speakers))[0]

        source_mcc = train_mcc[source_indices]
        source_labels = train_labels[source_indices]

        target_mcc = train_mcc[target_indices]
        target_labels = train_labels[target_indices]

        source_dataset = TensorDataset(source_mcc, source_labels)
        target_dataset = TensorDataset(target_mcc, target_labels)

        self.source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
        self.target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)

        eval_dataset = TensorDataset(eval_mcc, eval_labels)
        ref_dataset = TensorDataset(ref_mcc, ref_labels)

        self.eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        self.ref_loader = DataLoader(ref_dataset, batch_size=16, shuffle=False)

    def train(self, load_state = False, num_epochs = 7):
        if load_state:
            self.loadModel()
        print("Training model...")

        for epoch in range(num_epochs):
            for i, source_sample in enumerate(self.source_loader):
                source_mcc_batch, source_speaker_id_batch = source_sample
                batch_size = source_mcc_batch.size(0)

                for j in range(batch_size):
                    # Source MCC and speaker ID
                    source_mcc = source_mcc_batch[j].unsqueeze(0).to(self.device)
                    source_speaker_id = source_speaker_id_batch[j].unsqueeze(0).to(self.device)

                    # Get the source speaker embedding
                    source_speaker_label = self.train_dataset.labels[source_speaker_id.item()]
                    source_emb = self.train_dataset.speaker_emb[source_speaker_label]

                    # Select a random target speaker ID and get its embedding
                    target_speaker_id = torch.randint(0, self.num_target_speakers, (1,)).to(self.device)
                    target_speaker_label = self.train_dataset.labels[
                        target_speaker_id.item() + self.num_source_speakers]  # Target speakers start after source
                    target_emb = self.train_dataset.speaker_emb[target_speaker_label]

                    # Generate fake MCC for the target speaker
                    fake_mcc = self.generator(source_mcc, target_emb)

                    # Train the discriminator
                    real_validity = self.discriminator(source_mcc)
                    fake_validity = self.discriminator(fake_mcc.detach())

                    # Adversarial loss for real and fake samples
                    d_loss_real = self.criterion_adv(real_validity, torch.ones_like(real_validity))
                    d_loss_fake = self.criterion_adv(fake_validity, torch.zeros_like(fake_validity))
                    d_loss = (d_loss_real + d_loss_fake) / 2

                    # Update discriminator
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Train the generator and domain classifier
                    fake_validity = self.discriminator(fake_mcc)
                    g_adv_loss = self.criterion_adv(fake_validity, torch.ones_like(fake_validity))

                    # Cycle consistency loss (reconstruct source MCC from fake MCC)
                    rec_mcc = self.generator(fake_mcc, target_emb)
                    cycle_loss = self.criterion_cycle(rec_mcc, source_mcc)

                    # Identity loss (identity mapping should return same MCC using source embedding)
                    id_mcc = self.generator(source_mcc, source_emb)
                    id_loss = self.criterion_identity(id_mcc, source_mcc)

                    # Fix: Classification loss only for the target speakers
                    cls_loss = self.criterion_cls(self.domain_classifier(fake_mcc), target_speaker_id)

                    # Total generator loss
                    g_loss = g_adv_loss + cycle_loss * 10 + id_loss * 5 + cls_loss
                    if g_loss < self.top_score:
                        self.top_score = g_loss
                        self.saveModel()

                    # Update generator and domain classifier
                    self.g_optimizer.zero_grad()
                    self.c_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    self.c_optimizer.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, CLS Loss: {cls_loss.item():.4f}")

            print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")
            self.g_scheduler.step()
            self.d_scheduler.step()
            self.c_scheduler.step()
            print("min: ", self.top_score)

    def voiceToTarget(self, target_label, path_to_source_data):
        mean_norm_f0 = self.ref_dataset.getMeanNormLogf0ForSpeaker(target_label)
        mean_norm_f0 = mean_norm_f0.cpu().numpy().astype(np.float64)
        self.loadModel()
        self.generator.eval()
        mat_data = scipy.io.loadmat(path_to_source_data)
        norm_log_f0 = mat_data['norm_log_f0']
        mcc = mat_data['mcc']
        source_parameter = mat_data['source_parameter']
        original_mcc_size = mat_data['original_mcc_size'][0]
        x_old = np.linspace(0, 1, 512)
        x_new = np.linspace(0, 1, original_mcc_size[1])
        mean_norm_f0 = interp1d(x_old, mean_norm_f0, kind='linear')(x_new)
        print(original_mcc_size)# [36, 558] first dim is not complete


        x_old = np.linspace(0, 1, 512)
        x_new = np.linspace(0, 1, original_mcc_size[1])
        norm_log_f0 = interp1d(x_old, norm_log_f0, kind='linear')(x_new)
        norm_log_f0 = np.reshape(norm_log_f0, (norm_log_f0.shape[1],))


        target_emb = ad.getSpeakerEmbeddingFromLabel(target_label)
        target_emb = torch.tensor(target_emb, dtype=torch.float32).to(self.device)
        mcc_tensor = torch.tensor(mcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fake_mcc = self.generator(mcc_tensor, target_emb)

        fake_mcc = fake_mcc.squeeze(0).squeeze(0).cpu().numpy()
        u.plot_mcc_comparison(mcc, fake_mcc)

        x_old = np.linspace(0, 1, 512)
        x_new = np.linspace(0, 1, original_mcc_size[1])
        fake_mcc_interp = np.zeros((original_mcc_size[1], 36))

        # Interpolate each row of fake_mcc
        for i in range(fake_mcc.shape[1]):
            # Create the interpolation function
            interp_func = interp1d(x_old, fake_mcc[:, i], kind='linear', fill_value='extrapolate')

            # Apply the interpolation function to the new x values
            fake_mcc_interp[:, i] = interp_func(x_new)
        spectral_envelope = idct(fake_mcc_interp, type=2, axis=0, norm='ortho')
        spectral_envelope = np.exp(spectral_envelope)
        spectral_envelope = spectral_envelope.astype(np.float64)

        plt.subplot(1, 2, 2)
        plt.imshow(spectral_envelope, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Fake MCC')
        plt.xlabel('Time Frames')
        plt.ylabel('MCC Coefficients')
        plt.show()

        aperiodicity = np.array(source_parameter['aperiodicity'][0][0])
        aperiodicity = np.reshape(aperiodicity, (aperiodicity.shape[1], aperiodicity.shape[0]))
        x_old_freq = np.linspace(0, 1, aperiodicity.shape[1])
        x_new_freq = np.linspace(0, 1, 36)
        aperiodicity_downsampled = np.zeros((aperiodicity.shape[0], 36))
        for i in range(aperiodicity.shape[0]):
            interpolator = interp1d(x_old_freq, aperiodicity[i, :], kind='linear')
            aperiodicity_downsampled[i, :] = interpolator(x_new_freq)
        aperiodicity = np.ascontiguousarray(aperiodicity_downsampled)



        print(norm_log_f0.shape, spectral_envelope.shape, aperiodicity.shape)

        fs = 22050
        synthesized_wave = pw.synthesize(norm_log_f0, spectral_envelope, aperiodicity, fs)
        sf.write("output_wave.wav", synthesized_wave, fs)
# ways to check: compare original mcc and fake, add original pitch to check if better, may not work because of reshaping
# maybe avg f0 causes this noise, or log has to be unpacked
# cost function in my opinion sucks
# less time frames than in original

model = Model()
# model.train()
model.voiceToTarget("VCC2TF2", "reference_data/resized_audio/VCC2TF1/30002.wav.mat")