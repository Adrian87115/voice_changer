import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.ndimage import zoom
import generator as g
import discriminator as d
import audio_dataset as ad
import utils as u
import warnings
import matplotlib.pyplot as plt
import itertools
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

class Model():
    def __init__(self, source, target):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.getData(source, target)
        self.generator_xy = g.Generator().to(self.device)
        self.generator_yx = g.Generator().to(self.device)
        self.discriminator_x = d.Discriminator().to(self.device)
        self.discriminator_y = d.Discriminator().to(self.device)
        self.g_optimizer_xy = optim.Adam(self.generator_xy.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        self.g_optimizer_yx = optim.Adam(self.generator_yx.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        self.d_optimizer_x = optim.Adam(self.discriminator_x.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.d_optimizer_y = optim.Adam(self.discriminator_y.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.criterion_adv = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.g_scheduler_xy = StepLR(self.g_optimizer_xy, step_size = 10000, gamma = 0.1)
        self.g_scheduler_yx = StepLR(self.g_optimizer_yx, step_size = 10000, gamma = 0.1)
        self.d_scheduler_x = StepLR(self.d_optimizer_x, step_size = 10000, gamma = 0.1)
        self.d_scheduler_y = StepLR(self.d_optimizer_y, step_size = 10000, gamma = 0.1)
        self.top_score = float('inf')
        self.source = source
        self.target = target

    def saveModel(self):
        file_path = "saved_model4.pth"
        torch.save({
            'generator_xy_state_dict': self.generator_xy.state_dict(),
            'generator_yx_state_dict': self.generator_yx.state_dict(),
            'discriminator_x_state_dict': self.discriminator_x.state_dict(),
            'discriminator_y_state_dict': self.discriminator_y.state_dict(),
            'g_optimizer_xy_state_dict': self.g_optimizer_xy.state_dict(),
            'g_optimizer_yx_state_dict': self.g_optimizer_yx.state_dict(),
            'd_optimizer_x_state_dict': self.d_optimizer_x.state_dict(),
            'd_optimizer_y_state_dict': self.d_optimizer_y.state_dict(),
            'top_score': self.top_score}, file_path)

    def loadModel(self):
        file_path = "saved_model3.pth"
        checkpoint = torch.load(file_path)
        self.generator_xy.load_state_dict(checkpoint['generator_xy_state_dict'])
        self.generator_yx.load_state_dict(checkpoint['generator_yx_state_dict'])
        self.discriminator_x.load_state_dict(checkpoint['discriminator_x_state_dict'])
        self.discriminator_y.load_state_dict(checkpoint['discriminator_y_state_dict'])
        self.g_optimizer_xy.load_state_dict(checkpoint['g_optimizer_xy_state_dict'])
        self.g_optimizer_yx.load_state_dict(checkpoint['g_optimizer_yx_state_dict'])
        self.d_optimizer_x.load_state_dict(checkpoint['d_optimizer_x_state_dict'])
        self.d_optimizer_y.load_state_dict(checkpoint['d_optimizer_y_state_dict'])
        self.top_score = checkpoint['top_score']

    def reset_grad(self):
        self.g_optimizer_xy.zero_grad()
        self.g_optimizer_yx.zero_grad()
        self.d_optimizer_x.zero_grad()
        self.d_optimizer_y.zero_grad()

    def getData(self, source, target):
        path_train = "training_data/resized_audio" #first 8 are source, next 4 are target
        path_eval = "evaluation_data/resized_audio" #used to evaluate results, the output should be compared to the reference
        path_ref = "reference_data/resized_audio" #used to compare original to created by listening to them
        self.train_dataset = ad.AudioDataset(path_train, source, target)
        self.eval_dataset = ad.AudioDataset(path_eval, source, target)
        source_mcc = torch.stack([torch.tensor(x, dtype = torch.float32) for x in self.train_dataset.source_mcc])
        target_mcc = torch.stack([torch.tensor(x, dtype = torch.float32) for x in self.train_dataset.target_mcc])
        eval_source_mcc = torch.stack([torch.tensor(x, dtype = torch.float32) for x in self.eval_dataset.source_mcc])
        eval_target_mcc = torch.stack([torch.tensor(x, dtype = torch.float32) for x in self.eval_dataset.target_mcc])
        source_labels = torch.tensor([ad.getId(source)] * len(self.train_dataset.source_mcc), dtype = torch.long)
        target_labels = torch.tensor([ad.getId(target)] * len(self.train_dataset.target_mcc), dtype = torch.long)
        eval_source_labels = torch.tensor([ad.getId(source)] * len(self.eval_dataset.source_mcc), dtype = torch.long)
        eval_target_labels = torch.tensor([ad.getId(target)] * len(self.eval_dataset.target_mcc), dtype = torch.long)
        source_dataset = TensorDataset(source_mcc, source_labels)
        target_dataset = TensorDataset(target_mcc, target_labels)
        eval_source_dataset = TensorDataset(eval_source_mcc, eval_source_labels)
        eval_target_dataset = TensorDataset(eval_target_mcc, eval_target_labels)
        self.source_loader = DataLoader(source_dataset, batch_size = 1, shuffle = False)
        self.target_loader = DataLoader(target_dataset, batch_size = 1, shuffle = False)
        self.eval_source_loader = DataLoader(eval_source_dataset, batch_size = 1, shuffle = False)
        self.eval_target_loader = DataLoader(eval_target_dataset, batch_size = 1, shuffle = False)

    def train(self, load_state = False, num_epochs = 10):
        if load_state:
            self.loadModel()
        print("Training model...")
        cycle_loss_lambda = 10
        identity_loss_lambda = 5
        for epoch in range(num_epochs):
            target_iter = iter(self.target_loader)
            for i, source_sample in enumerate(self.source_loader):
                try:
                    target_sample = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_sample = next(target_iter)
                source_mcc_batch, _ = source_sample
                target_mcc_batch, _ = target_sample
                source_mcc_batch = torch.stack([torch.tensor(ad.getMccSlice(mcc)) for mcc in source_mcc_batch])
                target_mcc_batch = torch.stack([torch.tensor(ad.getMccSlice(mcc)) for mcc in target_mcc_batch])
                source_mcc_batch = source_mcc_batch.transpose(-1, -2)
                target_mcc_batch = target_mcc_batch.transpose(-1, -2)
                batch_size = source_mcc_batch.size(0)
                target_batch_size = target_mcc_batch.size(0)

                if target_batch_size < batch_size:
                    repeat_factor = (batch_size + target_batch_size - 1) // target_batch_size
                    target_mcc_batch = target_mcc_batch.repeat(repeat_factor, 1, 1)[:batch_size]
                elif target_batch_size > batch_size:
                    target_mcc_batch = target_mcc_batch[:batch_size]

                source_mcc_batch = source_mcc_batch.to(self.device)
                target_mcc_batch = target_mcc_batch.to(self.device)

                # Forward-inverse mapping: x -> y -> x'
                fake_y = self.generator_xy(source_mcc_batch)
                cycle_x = self.generator_yx(fake_y)
                # if i % 10 == 0:
                #     fake_y_np = fake_y.cpu().detach().numpy().squeeze(0)  # Remove unnecessary dimensions
                #     source_mcc_np = source_mcc_batch.cpu().detach().numpy().squeeze(0)  # Remove unnecessary dimensions
                #
                #     # Create a figure with two subplots
                #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
                #
                #     # Plot fake_y
                #     axs[0].imshow(fake_y_np, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
                #     axs[0].set_title('Fake Spectrogram')
                #     axs[0].set_xlabel('Time Frames')
                #     axs[0].set_ylabel('MCC Coefficients')
                #     plt.colorbar(axs[0].images[0], ax=axs[0], orientation='vertical')
                #
                #     # Plot source_mcc_batch
                #     axs[1].imshow(source_mcc_np, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
                #     axs[1].set_title('Source MCC Batch')
                #     axs[1].set_xlabel('Time Frames')
                #     axs[1].set_ylabel('MCC Coefficients')
                #     plt.colorbar(axs[1].images[0], ax=axs[1], orientation='vertical')
                #
                #     # Adjust layout
                #     plt.tight_layout()
                #     plt.show()
                # Inverse-forward mapping: y -> x -> y'
                fake_x = self.generator_yx(target_mcc_batch)
                cycle_y = self.generator_xy(fake_x)

                # Identity mapping: x -> x', y -> y'
                identity_x = self.generator_yx(source_mcc_batch)
                identity_y = self.generator_xy(target_mcc_batch)

                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)

                cycle_loss = self.criterion_cycle(source_mcc_batch, cycle_x) + self.criterion_cycle(target_mcc_batch, cycle_y)
                identity_loss = self.criterion_identity(source_mcc_batch, identity_x) + self.criterion_identity(target_mcc_batch, identity_y)

                generator_loss_xy = self.criterion_adv(d_fake_y, torch.ones_like(d_fake_y))
                generator_loss_yx = self.criterion_adv(d_fake_x, torch.ones_like(d_fake_x))

                generator_loss = generator_loss_xy + generator_loss_yx + cycle_loss_lambda * cycle_loss + identity_loss_lambda * identity_loss

                self.reset_grad()
                generator_loss.backward()

                self.g_optimizer_xy.step()
                self.g_optimizer_yx.step()

                d_real_x = self.discriminator_x(source_mcc_batch)
                d_real_y = self.discriminator_y(target_mcc_batch)

                generated_x = self.generator_yx(target_mcc_batch)
                d_fake_x = self.discriminator_x(generated_x)

                generated_y = self.generator_xy(source_mcc_batch)
                d_fake_y = self.discriminator_y(generated_y)

                d_loss_x_real = self.criterion_adv(d_real_x, torch.ones_like(d_real_x))
                d_loss_x_fake = self.criterion_adv(d_fake_x, torch.zeros_like(d_fake_x))
                d_loss_x = (d_loss_x_real + d_loss_x_fake) / 2.0

                d_loss_y_real = self.criterion_adv(d_real_y, torch.ones_like(d_real_y))
                d_loss_y_fake = self.criterion_adv(d_fake_y, torch.zeros_like(d_fake_y))
                d_loss_y = (d_loss_y_real + d_loss_y_fake) / 2.0

                d_loss = (d_loss_x + d_loss_y) / 2.0

                self.reset_grad()
                d_loss.backward()

                self.d_optimizer_x.step()
                self.d_optimizer_y.step()

                # if d_loss + generator_loss < self.top_score:
                #     self.top_score = d_loss + generator_loss

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {generator_loss.item():.4f}, "
                      f"Cycle Loss: {cycle_loss.item():.4f}")

            self.saveModel()
            avg_eval_loss_xy, avg_eval_loss_yx, avg_cycle_loss, avg_adv_loss = self.evaluate()
            print(f"xy loss: {avg_eval_loss_xy:.4f}, yx loss: {avg_eval_loss_yx:.4f}, Cycle loss: {avg_cycle_loss:.4f}, Adv loss: {avg_adv_loss:.4f}")

            self.g_scheduler_xy.step()
            self.g_scheduler_yx.step()
            self.d_scheduler_x.step()
            self.d_scheduler_y.step()
            print("Min total loss: ", self.top_score)

    def evaluate(self):
        self.loadModel()
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        avg_eval_loss_xy = 0.0
        avg_eval_loss_yx = 0.0
        avg_cycle_loss = 0.0
        avg_adv_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for eval_sample in self.eval_source_loader:
                eval_mcc_batch, eval_speaker_id_batch = eval_sample
                eval_mcc_batch = torch.stack([torch.tensor(ad.getMccSlice(mcc)) for mcc in eval_mcc_batch])
                eval_mcc_batch = eval_mcc_batch.transpose(-1, -2).to(self.device)

                fake_y = self.generator_xy(eval_mcc_batch)
                rec_x = self.generator_yx(fake_y)

                cycle_loss_x = self.criterion_cycle(rec_x, eval_mcc_batch)

                fake_validity_y = self.discriminator_x(fake_y)
                adv_loss_xy = self.criterion_adv(fake_validity_y, torch.ones_like(fake_validity_y))

                avg_eval_loss_xy += adv_loss_xy.item()
                avg_cycle_loss += cycle_loss_x.item()
                avg_adv_loss += adv_loss_xy.item()

                num_batches += 1

            for eval_sample in self.eval_target_loader:
                eval_mcc_batch, eval_speaker_id_batch = eval_sample
                eval_mcc_batch = torch.stack([torch.tensor(ad.getMccSlice(mcc)) for mcc in eval_mcc_batch])
                eval_mcc_batch = eval_mcc_batch.transpose(-1, -2).to(self.device)

                fake_x = self.generator_yx(eval_mcc_batch)
                rec_y = self.generator_xy(fake_x)

                cycle_loss_y = self.criterion_cycle(rec_y, eval_mcc_batch)

                fake_validity_x = self.discriminator_x(fake_x)
                adv_loss_yx = self.criterion_adv(fake_validity_x, torch.ones_like(fake_validity_x))

                avg_eval_loss_yx += adv_loss_yx.item()
                avg_cycle_loss += cycle_loss_y.item()
                avg_adv_loss += adv_loss_yx.item()
                num_batches += 1
        avg_eval_loss_xy /= num_batches
        avg_eval_loss_yx /= num_batches
        avg_cycle_loss /= num_batches
        avg_adv_loss /= num_batches
        return avg_eval_loss_xy, avg_eval_loss_yx, avg_cycle_loss, avg_adv_loss

    def voiceToTarget(self, source, target, path_to_source_data):
        self.loadModel()
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        data = np.load(path_to_source_data)
        norm_log_f0 = data['norm_log_f0']
        mean_log_f0 = data['mean_log_f0']
        std_log_f0 = data['std_log_f0']
        mcc = data['mcc'].T
        ap = data['source_parameter']
        tf = data['time_frames']
        norm_log_f0_zoom_factor = (tf / norm_log_f0.size,)
        norm_log_f0 = zoom(norm_log_f0, norm_log_f0_zoom_factor, order = 1)
        log_f0 = mean_log_f0 + std_log_f0 * norm_log_f0
        f0_target = np.exp(log_f0) - 1e-5

        chunk_size = 128
        num_chunks = (mcc.shape[1] + chunk_size - 1) // chunk_size
        fake_mcc_chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, mcc.shape[1])
            mcc_chunk = mcc[:, start:end]
            mcc_tensor = torch.tensor(mcc_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if source == self.source and target == self.target:
                    fake_mcc_chunk = self.generator_xy(mcc_tensor)
                    is_real = self.discriminator_y(fake_mcc_chunk)
                elif source == self.target and target == self.source:
                    fake_mcc_chunk = self.generator_yx(mcc_tensor)
                    is_real = self.discriminator_x(fake_mcc_chunk)
                else:
                    raise ValueError("Invalid pair, current model is not for the selected source and target")
                fake_mcc_chunks.append(fake_mcc_chunk.squeeze(0).cpu().numpy())
                print(f"Probability synthesized audio is real for chunk {i}: {is_real[0].item():.4f}")
        fake_mcc = np.concatenate(fake_mcc_chunks, axis=1)
        fake_mcc = fake_mcc.T
        mcc_zoom_factor = (tf / fake_mcc.shape[0], 1)
        fake_mcc = zoom(fake_mcc, mcc_zoom_factor, order=1).astype(np.float64)
        synthesized_wav = u.reassembleWav(f0_target, fake_mcc, ap, 22050, 5)
        u.saveWav(synthesized_wav, "out_synthesized.wav", 22050)

        # plt.plot(f0_target)
        # plt.title('Pitch Contour (f0)')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Pitch (Hz)')
        # plt.grid(True)
        # plt.show()

        plt.imshow(fake_mcc.T, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        plt.colorbar()
        plt.title('Fake Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCC Coefficients')
        plt.show()

        plt.imshow(mcc, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        plt.colorbar()
        plt.title('Original Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCC Coefficients')
        plt.show()

        # plt.imshow(ap, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        # plt.colorbar()
        # plt.title('Aperiodicity')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Value')
        # plt.show()