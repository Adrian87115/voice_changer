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
        file_path = "saved_model2.pth"
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'domain_classifier_state_dict': self.domain_classifier.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'c_optimizer_state_dict': self.c_optimizer.state_dict(),
            'top_score': self.top_score}, file_path)

    def loadModel(self):
        file_path = "saved_model2.pth"
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

    def train(self, load_state = False, num_epochs = 7):
        if load_state:
            self.loadModel()
        print("Training model...")
        target_iter = itertools.cycle(self.target_loader)

        for epoch in range(num_epochs):
            for i, source_sample in enumerate(self.source_loader):
                target_sample = next(target_iter)
                source_mcc_batch, source_speaker_id_batch = source_sample
                target_mcc_batch, target_speaker_id_batch = target_sample
                batch_size = source_mcc_batch.size(0)
                target_batch_size = target_mcc_batch.size(0)


                if target_batch_size < batch_size:
                    repeat_factor = (batch_size + target_batch_size - 1) // target_batch_size
                    target_mcc_batch = target_mcc_batch.repeat(repeat_factor, 1, 1)[:batch_size]
                    target_speaker_id_batch = target_speaker_id_batch.repeat(repeat_factor)[:batch_size]

                total_d_loss, total_g_loss, total_dc_loss, total_cycle_loss = 0, 0, 0, 0
                for j in range(batch_size):

                    #1 source mcc + choosen target label to generator:
                    source_mcc = source_mcc_batch[j].unsqueeze(0).to(self.device)
                    source_speaker_id = source_speaker_id_batch[j].unsqueeze(0).to(self.device)
                    source_speaker_label = self.train_dataset.labels[source_speaker_id.item()]
                    source_emb = self.train_dataset.one_hot_labels[source_speaker_label]
                    target_speaker_id_to_convert = torch.randint(0, self.num_target_speakers, (1,)).to(self.device)
                    target_speaker_label_to_convert = ad.target_labels[target_speaker_id_to_convert]
                    target_emb_to_convert = self.train_dataset.one_hot_labels[target_speaker_label_to_convert]
                    fake_mcc = self.generator(source_mcc, target_emb_to_convert)

                    #2 fake mcc + source original label to generator:
                    rec_mcc = self.generator(fake_mcc, source_emb)
                    cycle_loss = self.criterion_cycle(rec_mcc, source_mcc)

                    #3 fake mcc to classifier:
                    fake_class_pred = self.domain_classifier(fake_mcc)
                    dc_loss_fake = self.criterion_cls(fake_class_pred, target_speaker_id_to_convert)

                    #4 target real mcc to classifier:
                    target_real_mcc = target_mcc_batch[j].unsqueeze(0).to(self.device)
                    target_real_speaker_id = target_speaker_id_batch[j].unsqueeze(0).to(self.device)
                    target_real_speaker_label = ad.all_labels[target_real_speaker_id]
                    target_real_emb = ad.getSpeakerOneHotFromLabel(target_real_speaker_label)
                    target_real_speaker_id = target_real_speaker_id - 8
                    target_class_pred = self.domain_classifier(target_real_mcc)
                    dc_loss_real = self.criterion_cls(target_class_pred, target_real_speaker_id)

                    #5 fake mcc + choosen target label to discriminator:
                    validity_fake = self.discriminator(fake_mcc, target_emb_to_convert[8:])
                    d_loss_fake = self.criterion_adv(validity_fake, torch.zeros_like(validity_fake))

                    #6 target real mcc + target real label to discriminator:
                    validity_real = self.discriminator(target_real_mcc, target_real_emb[8:])
                    d_loss_real = self.criterion_adv(validity_real, torch.ones_like(validity_real))


                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()
                    self.c_optimizer.zero_grad()
                    g_loss = self.criterion_adv(validity_fake, torch.ones_like(validity_fake)) + cycle_loss
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    dc_loss = (dc_loss_fake + dc_loss_real) / 2
                    g_loss.backward(retain_graph=True)
                    d_loss.backward(retain_graph=True)
                    dc_loss.backward()
                    self.g_optimizer.step()
                    self.d_optimizer.step()
                    self.c_optimizer.step()


                    total_cycle_loss += cycle_loss.item()
                    total_dc_loss += (dc_loss_fake + dc_loss_real).item() / 2
                    total_d_loss += (d_loss_real + d_loss_fake).item() / 2
                    total_g_loss += self.criterion_adv(validity_fake, torch.ones_like(validity_fake)).item()
                    total_loss = total_cycle_loss + total_dc_loss + total_d_loss + total_g_loss
                    if total_loss < self.top_score:
                        self.top_score = total_loss
                        self.saveModel()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"D Loss: {total_d_loss / batch_size:.4f}, G Loss: {total_g_loss / batch_size:.4f}, CLS Loss: {total_dc_loss / batch_size:.4f}, Cycle Loss: {total_cycle_loss / batch_size:.4f}")

            eval_loss, accuracy = self.evaluate()
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Evaluation Loss: {eval_loss:.4f}, Accuracy: {accuracy:.2f}%")
            self.g_scheduler.step()
            self.d_scheduler.step()
            self.c_scheduler.step()
            print("min: ", self.top_score)

    def evaluate(self):
        self.generator.eval()
        self.domain_classifier.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, eval_sample in enumerate(self.eval_loader):
                eval_mcc_batch, eval_speaker_id_batch = eval_sample
                batch_size = eval_mcc_batch.size(0)
                for j in range(batch_size):
                    eval_mcc = eval_mcc_batch[j].unsqueeze(0).to(self.device)
                    target_speaker_id = torch.randint(0, self.num_target_speakers, (1,)).to(self.device)
                    target_speaker_label = ad.target_labels[target_speaker_id]
                    target_emb = self.train_dataset.one_hot_labels[target_speaker_label]
                    fake_mcc = self.generator(eval_mcc, target_emb)
                    predicted_speaker_logits = self.domain_classifier(fake_mcc)
                    cls_loss = self.criterion_cls(predicted_speaker_logits, target_speaker_id)
                    eval_loss += cls_loss.item()
                    predicted_speaker_id = torch.argmax(predicted_speaker_logits, dim=1)
                    correct += (predicted_speaker_id == target_speaker_id).sum().item()
                    total += batch_size
        avg_eval_loss = eval_loss / total
        accuracy = correct / total * 100
        return avg_eval_loss, accuracy

    def voiceToTarget(self, target_label, path_to_source_data):
        self.loadModel()
        self.generator.eval()
        data = np.load(path_to_source_data)
        norm_log_f0 = data['norm_log_f0']
        mean_log_f0 = data['mean_log_f0']
        std_log_f0 = data['std_log_f0']
        mcc = data['mcc']
        ap = data['source_parameter']
        tf = data['time_frames']

        norm_log_f0_zoom_factor = (tf / norm_log_f0.size,)

        norm_log_f0 = zoom(norm_log_f0, norm_log_f0_zoom_factor, order=1)
        log_f0 = mean_log_f0 + std_log_f0 * norm_log_f0
        f0_target = np.exp(log_f0) - 1e-5

        target_emb = ad.getSpeakerOneHotFromLabel(target_label)
        target_emb = torch.tensor(target_emb, dtype=torch.float32).to(self.device)
        mcc_tensor = torch.tensor(mcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fake_mcc = self.generator(mcc_tensor, target_emb)
            predicted_speaker_logits = self.domain_classifier(fake_mcc)
            predicted_speaker_id = torch.argmax(predicted_speaker_logits, dim=1).item()
            predicted_speaker_label = ad.target_labels[predicted_speaker_id]
            print(f"Synthesized audio is predicted to correspond to speaker: {predicted_speaker_label}")
            target_emb = self.train_dataset.one_hot_labels[target_label]
            is_true = self.discriminator(fake_mcc, target_emb[8:])
            print(f"Probability synthesized audio is real: {is_true[0].item():.4f}")
        fake_mcc = fake_mcc.squeeze(0).squeeze(0)
        fake_mcc = fake_mcc.cpu().numpy()

        mcc_zoom_factor = (tf / fake_mcc.shape[0], 1)
        fake_mcc = zoom(fake_mcc, mcc_zoom_factor, order=1)
        fake_mcc = fake_mcc.astype(np.float64)

        synthesized_wav = u.reassembleWav(f0_target, fake_mcc, ap, 22050, 5)
        u.saveWav(synthesized_wav, "out_synthesized.wav", 22050)

        plt.plot(f0_target)
        plt.title('Pitch Contour (f0)')
        plt.xlabel('Time Frames')
        plt.ylabel('Pitch (Hz)')
        plt.grid(True)
        plt.show()

        plt.imshow(fake_mcc, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
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
        plt.ylabel('val')
        plt.show()