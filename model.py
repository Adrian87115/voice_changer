import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.ndimage import zoom
import generator as g
import discriminator as d
import domain_classifier as dc
import audio_dataset as ad
import utils as u
import warnings
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as f
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")
# ATTRIBUTE IS THE LABEL!!!!!!!!
class Model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_source_speakers = 8
        self.num_target_speakers = 4
        self.getData()
        self.generator = g.Generator().to(self.device)
        self.discriminator = d.Discriminator().to(self.device)
        self.domain_classifier = dc.DomainClassifier().to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.criterion_adv = f.binary_cross_entropy_with_logits
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = f.binary_cross_entropy_with_logits
        self.criterion_cls = nn.CrossEntropyLoss()
        self.g_scheduler = StepLR(self.g_optimizer, step_size=1, gamma=0.1)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=1, gamma=0.1)
        self.c_scheduler = StepLR(self.c_optimizer, step_size=1, gamma=0.1)
        self.top_score = float('inf')
        self.n_critic = 3

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
        path_eval = "evaluation_data/resized_audio" #used to evaluate results, the output should be compared to the reference
        path_ref = "reference_data/resized_audio" #used to compare original to created by listening to them
        self.train_dataset = ad.AudioDataset(path_train)
        self.eval_dataset = ad.AudioDataset(path_eval)
        self.ref_dataset = ad.AudioDataset(path_ref)

        train_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.train_dataset.mcc])
        eval_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.eval_dataset.mcc])
        ref_mcc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.ref_dataset.mcc])

        train_labels = torch.tensor([ad.getId(label) for label in self.train_dataset.labels], dtype=torch.long)
        eval_labels = torch.tensor([ad.getId(label) for label in self.eval_dataset.labels], dtype=torch.long)
        ref_labels = torch.tensor([ad.getId(label) for label in self.ref_dataset.labels], dtype=torch.long)
        eval_dataset = TensorDataset(eval_mcc, eval_labels)
        ref_dataset = TensorDataset(ref_mcc, ref_labels)

        source_indices = torch.where(train_labels < self.num_source_speakers)[0]
        target_indices = torch.where((train_labels >= self.num_source_speakers) & (train_labels < self.num_source_speakers + self.num_target_speakers))[0]
        source_mcc = train_mcc[source_indices]
        source_labels = train_labels[source_indices]
        target_mcc = train_mcc[target_indices]
        target_labels = train_labels[target_indices]
        source_dataset = TensorDataset(source_mcc, source_labels)
        target_dataset = TensorDataset(target_mcc, target_labels)
        self.source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
        self.target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        self.ref_loader = DataLoader(ref_dataset, batch_size=16, shuffle=False)

    def train(self, load_state = False, num_epochs = 1):
        if load_state:
            self.loadModel()
        print("Training model...")

        for epoch in range(num_epochs):
            target_iter = iter(self.target_loader)
            for i, source_sample in enumerate(self.source_loader):
                try:
                    target_sample = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_sample = next(target_iter)
                source_mcc_batch, source_speaker_id_batch = source_sample
                target_mcc_batch, target_speaker_id_batch = target_sample
                source_mcc_batch = source_mcc_batch.transpose(-1, -2)
                target_mcc_batch = target_mcc_batch.transpose(-1, -2)

                batch_size = source_mcc_batch.size(0)
                target_batch_size = target_mcc_batch.size(0)

                if target_batch_size < batch_size:
                    repeat_factor = (batch_size + target_batch_size - 1) // target_batch_size
                    target_mcc_batch = target_mcc_batch.repeat(repeat_factor, 1, 1)[:batch_size]
                    target_speaker_id_batch = target_speaker_id_batch.repeat(repeat_factor)[:batch_size]
                elif target_batch_size > batch_size:
                    target_mcc_batch = target_mcc_batch[:batch_size]
                    target_speaker_id_batch = target_speaker_id_batch[:batch_size]

                source_mcc_batch = source_mcc_batch.to(self.device)
                source_speaker_id_batch = source_speaker_id_batch.to(self.device)
                target_mcc_batch = target_mcc_batch.to(self.device)
                target_speaker_id_batch = target_speaker_id_batch.to(self.device)

                source_speaker_labels = [ad.all_labels[id.item()] for id in source_speaker_id_batch]
                source_emb_batch = torch.stack([self.train_dataset.one_hot_labels[label] for label in source_speaker_labels])

                target_speaker_id_batch = target_speaker_id_batch - 8
                target_speaker_labels = [ad.target_labels[id.item()] for id in target_speaker_id_batch]
                target_emb_batch = torch.stack([self.train_dataset.one_hot_labels[label] for label in target_speaker_labels])

                # use n_critic

                # 1. Source MCC + target label to generator
                fake_mcc_batch = self.generator(source_mcc_batch, target_emb_batch)
                # for idx, fake_mcc in enumerate(fake_mcc_batch):
                #     fake_mcc = fake_mcc.detach().squeeze(0).cpu().numpy()
                #     plt.figure(figsize=(10, 4))
                #     plt.imshow(fake_mcc, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
                #     plt.colorbar()
                #     plt.title(f'Fake MCC - Sample {idx + 1}')
                #     plt.xlabel('Time Frames')
                #     plt.ylabel('MCC Coefficients')
                #     plt.show()
                #     break

                # 2. Fake MCC + source label to generator (Cycle consistency loss)
                rec_mcc_batch = self.generator(fake_mcc_batch, source_emb_batch)

                # plt.imshow(rec_mcc_batch[0].detach().squeeze(0).cpu().numpy(), aspect='auto', origin='lower',
                #            cmap='viridis', interpolation='none')
                # plt.colorbar()
                # plt.xlabel('Time Frames')
                # plt.ylabel('MCC Coefficients')
                # plt.show()
                # plt.imshow(fake_mcc_batch[0].detach().squeeze(0).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis', interpolation='none')
                # plt.colorbar()
                # plt.xlabel('Time Frames')
                # plt.ylabel('MCC Coefficients')
                # plt.show()
                cycle_loss = self.criterion_cycle(rec_mcc_batch, source_mcc_batch)

                # 3. Fake MCC to domain classifier
                fake_class_pred_batch = self.domain_classifier(fake_mcc_batch)
                dc_loss_fake = self.criterion_cls(fake_class_pred_batch, target_speaker_id_batch)

                # 4. Real target MCC to domain classifier
                target_class_pred_batch = self.domain_classifier(target_mcc_batch)
                dc_loss_real = self.criterion_cls(target_class_pred_batch, target_speaker_id_batch)

                # 5. Fake MCC + chosen target label to discriminator
                validity_fake_batch = self.discriminator(fake_mcc_batch, target_emb_batch[:, 8:])
                d_loss_fake = self.criterion_adv(validity_fake_batch, torch.zeros_like(validity_fake_batch))

                # 6. Real target MCC + real target label to discriminator
                validity_real_batch = self.discriminator(target_mcc_batch, target_emb_batch[:, 8:])
                d_loss_real = self.criterion_adv(validity_real_batch, torch.ones_like(validity_real_batch))

                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                self.c_optimizer.zero_grad()

                g_loss = self.criterion_adv(validity_fake_batch, torch.ones_like(validity_fake_batch)) + cycle_loss
                d_loss = d_loss_fake + 2 * d_loss_real
                dc_loss = dc_loss_fake + 2 * dc_loss_real
                total_loss = g_loss + d_loss + dc_loss
                total_loss.backward()

                self.g_optimizer.step()
                self.d_optimizer.step()
                self.c_optimizer.step()

                if total_loss < self.top_score:
                    self.top_score = total_loss
                    self.saveModel()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"CLS Loss: {dc_loss.item():.4f}, Cycle Loss: {cycle_loss.item():.4f}")

            eval_loss, accuracy = self.evaluate()
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Evaluation Loss: {eval_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.g_scheduler.step()
            self.d_scheduler.step()
            self.c_scheduler.step()
            print("Min total loss: ", self.top_score)

    def evaluate(self):#make better eval
        self.loadModel()
        self.generator.eval()
        self.discriminator.eval()
        self.domain_classifier.eval()

        eval_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        with torch.no_grad():
            for eval_sample in self.eval_loader:
                eval_mcc_batch, eval_speaker_id_batch = eval_sample
                eval_mcc_batch = eval_mcc_batch.transpose(-1, -2)
                batch_size = eval_mcc_batch.size(0)
                n_batches += 1
                eval_mcc_batch = eval_mcc_batch.to(self.device)
                target_speaker_id_batch = torch.randint(0, self.num_target_speakers, (batch_size,)).to(self.device)
                target_speaker_labels_batch = [ad.target_labels[id.item()] for id in target_speaker_id_batch]
                target_emb_batch = torch.stack([self.train_dataset.one_hot_labels[label] for label in target_speaker_labels_batch])
                fake_mcc_batch = self.generator(eval_mcc_batch, target_emb_batch)
                predicted_speaker_logits_batch = self.domain_classifier(fake_mcc_batch)
                cls_loss = self.criterion_cls(predicted_speaker_logits_batch, target_speaker_id_batch)
                eval_loss += cls_loss.item()

                predicted_speaker_id_batch = torch.argmax(predicted_speaker_logits_batch, dim=1)
                correct += (predicted_speaker_id_batch == target_speaker_id_batch).sum().item()
                total += batch_size

        avg_eval_loss = eval_loss / n_batches
        accuracy = correct / total * 100

        return avg_eval_loss, accuracy

    def voiceToTarget(self, target_label, path_to_source_data):
        self.loadModel()
        self.generator.eval()
        self.domain_classifier.eval()
        self.discriminator.eval()

        data = np.load(path_to_source_data)
        norm_log_f0 = data['norm_log_f0']
        mean_log_f0 = data['mean_log_f0']
        std_log_f0 = data['std_log_f0']
        mcc = data['mcc'].T
        ap = data['source_parameter']
        tf = data['time_frames']
        norm_log_f0_zoom_factor = (tf / norm_log_f0.size,)
        norm_log_f0 = zoom(norm_log_f0, norm_log_f0_zoom_factor, order=1)
        log_f0 = mean_log_f0 + std_log_f0 * norm_log_f0
        f0_target = np.exp(log_f0) - 1e-5
        target_emb = ad.getSpeakerOneHotFromLabel(target_label)
        target_emb = torch.tensor(target_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        mcc_tensor = torch.tensor(mcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fake_mcc = self.generator(mcc_tensor, target_emb)
            predicted_speaker_logits = self.domain_classifier(fake_mcc)
            predicted_speaker_id = torch.argmax(predicted_speaker_logits, dim=1).item()
            predicted_speaker_label = ad.target_labels[predicted_speaker_id]
            predicted_speaker_prob = torch.softmax(predicted_speaker_logits, dim=1)
            predicted_speaker_prob = predicted_speaker_prob[0, predicted_speaker_id].item()
            print(f"Synthesized audio is predicted to correspond to speaker: {predicted_speaker_label}, with probability: {predicted_speaker_prob:.4f}")
            is_real = self.discriminator(fake_mcc, target_emb[:, 8:])
            print(f"Probability synthesized audio is real: {is_real[0].item():.4f}")
        fake_mcc = fake_mcc.squeeze(0).squeeze(0).cpu().numpy().T
        mcc_zoom_factor = (tf / fake_mcc.shape[0], 1)
        fake_mcc = zoom(fake_mcc, mcc_zoom_factor, order=1).astype(np.float64)
        synthesized_wav = u.reassembleWav(f0_target, fake_mcc, ap, 22050, 5)
        u.saveWav(synthesized_wav, "out_synthesized.wav", 22050)

        plt.plot(f0_target)
        plt.title('Pitch Contour (f0)')
        plt.xlabel('Time Frames')
        plt.ylabel('Pitch (Hz)')
        plt.grid(True)
        plt.show()

        plt.imshow(fake_mcc.T, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Fake Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCC Coefficients')
        plt.show()

        plt.imshow(mcc, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Original Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCC Coefficients')
        plt.show()

        plt.imshow(ap, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Aperiodicity')
        plt.xlabel('Time Frames')
        plt.ylabel('Value')
        plt.show()

# try approach with residual blocks from stargan vc2