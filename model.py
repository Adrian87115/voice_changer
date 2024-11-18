import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
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
        self.source = source
        self.target = target
        self.iteration = 0

    def saveModel(self, epoch):
        file_path = f"saved_model_epoch_{epoch}.pth"
        torch.save({
            'generator_xy_state_dict': self.generator_xy.state_dict(),
            'generator_yx_state_dict': self.generator_yx.state_dict(),
            'discriminator_x_state_dict': self.discriminator_x.state_dict(),
            'discriminator_y_state_dict': self.discriminator_y.state_dict(),
            'g_optimizer_xy_state_dict': self.g_optimizer_xy.state_dict(),
            'g_optimizer_yx_state_dict': self.g_optimizer_yx.state_dict(),
            'd_optimizer_x_state_dict': self.d_optimizer_x.state_dict(),
            'd_optimizer_y_state_dict': self.d_optimizer_y.state_dict(),
            'iteration': self.iteration}, file_path)

    def loadModel(self):
        file_path = "saved_model5_init325.pth"
        checkpoint = torch.load(file_path)
        self.generator_xy.load_state_dict(checkpoint['generator_xy_state_dict'])
        self.generator_yx.load_state_dict(checkpoint['generator_yx_state_dict'])
        self.discriminator_x.load_state_dict(checkpoint['discriminator_x_state_dict'])
        self.discriminator_y.load_state_dict(checkpoint['discriminator_y_state_dict'])
        self.g_optimizer_xy.load_state_dict(checkpoint['g_optimizer_xy_state_dict'])
        self.g_optimizer_yx.load_state_dict(checkpoint['g_optimizer_yx_state_dict'])
        self.d_optimizer_x.load_state_dict(checkpoint['d_optimizer_x_state_dict'])
        self.d_optimizer_y.load_state_dict(checkpoint['d_optimizer_y_state_dict'])
        self.iteration = checkpoint['iteration']

    def resetGrad(self):
        self.g_optimizer_xy.zero_grad()
        self.g_optimizer_yx.zero_grad()
        self.d_optimizer_x.zero_grad()
        self.d_optimizer_y.zero_grad()

    def getData(self, source, target):
        path_train = "training_data/transformed_audio" #first 8 are source, next 4 are target
        path_eval = "evaluation_data/transformed_audio" #used to evaluate results, the output should be compared to the reference
        path_ref = "reference_data/transformed_audio" #used to compare original to created by listening to them
        self.train_dataset = ad.AudioDataset(path_train, source, target)
        self.eval_dataset = ad.AudioDataset(path_eval, source, target)

        source_dataset = u.MCEPDataset(self.train_dataset.source_mcep, ad.getId(source))
        target_dataset = u.MCEPDataset(self.train_dataset.target_mcep, ad.getId(target))
        eval_source_dataset = u.MCEPDataset(self.eval_dataset.source_mcep, ad.getId(source))
        eval_target_dataset = u.MCEPDataset(self.eval_dataset.target_mcep, ad.getId(target))

        self.source_loader = DataLoader(source_dataset, batch_size = 1, shuffle = False)
        self.target_loader = DataLoader(target_dataset, batch_size = 1, shuffle = False)
        self.eval_source_loader = DataLoader(eval_source_dataset, batch_size = 1, shuffle = False)
        self.eval_target_loader = DataLoader(eval_target_dataset, batch_size = 1, shuffle = False)

    def train(self, load_state = False, num_epochs = 100):
        if load_state:
            self.loadModel()
        print("Training model...")
        cycle_loss_lambda = 10
        identity_loss_lambda = 5
        for epoch in range(num_epochs):
            target_iter = iter(self.target_loader)
            for i, source_sample in enumerate(self.source_loader):
                self.iteration += 1
                if self.iteration > 10000:
                    identity_loss_lambda = 0
                try:
                    target_sample = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_sample = next(target_iter)
                source_mcep_batch, _ = source_sample
                target_mcep_batch, _ = target_sample
                source_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in source_mcep_batch])
                target_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in target_mcep_batch])
                source_mcep_batch = source_mcep_batch.transpose(-1, -2)
                target_mcep_batch = target_mcep_batch.transpose(-1, -2)
                batch_size = source_mcep_batch.size(0)
                target_batch_size = target_mcep_batch.size(0)
                if target_batch_size < batch_size:
                    repeat_factor = (batch_size + target_batch_size - 1) // target_batch_size
                    target_mcep_batch = target_mcep_batch.repeat(repeat_factor, 1, 1)[:batch_size]
                elif target_batch_size > batch_size:
                    target_mcep_batch = target_mcep_batch[:batch_size]

                source_mcep_batch = source_mcep_batch.to(self.device)
                target_mcep_batch = target_mcep_batch.to(self.device)

                # Forward-inverse mapping: x -> y -> x'
                fake_y = self.generator_xy(source_mcep_batch)
                cycle_x = self.generator_yx(fake_y)

                # Inverse-forward mapping: y -> x -> y'
                fake_x = self.generator_yx(target_mcep_batch)
                cycle_y = self.generator_xy(fake_x)

                # Identity mapping: x -> x', y -> y'
                identity_x = self.generator_yx(source_mcep_batch)
                identity_y = self.generator_xy(target_mcep_batch)

                # Adversarial loss for direct mappings (first step)
                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)
                generator_loss_xy = self.criterion_adv(d_fake_y, torch.ones_like(d_fake_y))
                generator_loss_yx = self.criterion_adv(d_fake_x, torch.ones_like(d_fake_x))

                # Adversarial loss for cycle-consistent mappings (second step)
                d_cycle_x = self.discriminator_x(cycle_x)
                d_cycle_y = self.discriminator_y(cycle_y)
                generator_loss_cycle_x = self.criterion_adv(d_cycle_x, torch.ones_like(d_cycle_x))
                generator_loss_cycle_y = self.criterion_adv(d_cycle_y, torch.ones_like(d_cycle_y))

                # Cycle and identity consistency losses
                cycle_loss = self.criterion_cycle(source_mcep_batch, cycle_x) + self.criterion_cycle(target_mcep_batch, cycle_y)
                identity_loss = self.criterion_identity(source_mcep_batch, identity_x) + self.criterion_identity(target_mcep_batch, identity_y)

                generator_loss = generator_loss_xy + generator_loss_yx + generator_loss_cycle_x + generator_loss_cycle_y + cycle_loss_lambda * cycle_loss + identity_loss_lambda * identity_loss

                self.resetGrad()
                generator_loss.backward()
                self.g_optimizer_xy.step()
                self.g_optimizer_yx.step()

                # Discriminator losses for source and target domains
                d_real_x = self.discriminator_x(source_mcep_batch)
                d_real_y = self.discriminator_y(target_mcep_batch)

                # Fake samples for discriminator training
                generated_x = self.generator_yx(target_mcep_batch).detach()
                generated_y = self.generator_xy(source_mcep_batch).detach()

                # Discriminator loss for X
                d_fake_x = self.discriminator_x(generated_x)
                d_loss_x_real = self.criterion_adv(d_real_x, torch.ones_like(d_real_x))
                d_loss_x_fake = self.criterion_adv(d_fake_x, torch.zeros_like(d_fake_x))
                d_loss_x = (d_loss_x_real + d_loss_x_fake) / 2.0

                # Discriminator loss for Y
                d_fake_y = self.discriminator_y(generated_y)
                d_loss_y_real = self.criterion_adv(d_real_y, torch.ones_like(d_real_y))
                d_loss_y_fake = self.criterion_adv(d_fake_y, torch.zeros_like(d_fake_y))
                d_loss_y = (d_loss_y_real + d_loss_y_fake) / 2.0

                d_loss = (d_loss_x + d_loss_y) / 2.0

                self.resetGrad()
                d_loss.backward()
                self.d_optimizer_x.step()
                self.d_optimizer_y.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {generator_loss.item():.4f}, "
                      f"Cycle Loss: {cycle_loss.item():.4f}")

            if (epoch + 1) % 20 == 0:
                self.saveModel(epoch + 1)

            # avg_eval_loss_xy, avg_eval_loss_yx, avg_cycle_loss, avg_adv_loss = self.evaluate()
            # print(f"xy loss: {avg_eval_loss_xy:.4f}, yx loss: {avg_eval_loss_yx:.4f}, Cycle loss: {avg_cycle_loss:.4f}, Adv loss: {avg_adv_loss:.4f}")

            self.g_scheduler_xy.step()
            self.g_scheduler_yx.step()
            self.d_scheduler_x.step()
            self.d_scheduler_y.step()

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
                eval_mcep_batch, eval_speaker_id_batch = eval_sample
                eval_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in eval_mcep_batch])
                eval_mcep_batch = eval_mcep_batch.transpose(-1, -2).to(self.device)

                fake_y = self.generator_xy(eval_mcep_batch)
                rec_x = self.generator_yx(fake_y)

                cycle_loss_x = self.criterion_cycle(rec_x, eval_mcep_batch)

                fake_validity_y = self.discriminator_x(fake_y)
                adv_loss_xy = self.criterion_adv(fake_validity_y, torch.ones_like(fake_validity_y))

                avg_eval_loss_xy += adv_loss_xy.item()
                avg_cycle_loss += cycle_loss_x.item()
                avg_adv_loss += adv_loss_xy.item()

                num_batches += 1

            for eval_sample in self.eval_target_loader:
                eval_mcep_batch, eval_speaker_id_batch = eval_sample
                eval_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in eval_mcep_batch])
                eval_mcep_batch = eval_mcep_batch.transpose(-1, -2).to(self.device)

                fake_x = self.generator_yx(eval_mcep_batch)
                rec_y = self.generator_xy(fake_x)

                cycle_loss_y = self.criterion_cycle(rec_y, eval_mcep_batch)

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
        log_f0 = data['log_f0']
        mcep = data['mcep'].T

        if source == self.source:
            mcep = self.train_dataset.normalizeMcep(mcep, True)
        elif source == self.target:
            mcep = self.train_dataset.normalizeMcep(mcep, False)
        else:
            raise ValueError("Invalid pair, current model is not for the selected source and target")

        ap = data['source_parameter']
        tf = data['time_frames']
        pitch_dataset = ad.PitchDataset("evaluation_data/transformed_audio", source, target)
        f0_converted = pitch_dataset.pitchConversion(log_f0)
        chunk_size = 128
        num_chunks = (mcep.shape[1] + chunk_size - 1) // chunk_size
        fake_mcep_chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, mcep.shape[1])
            mcep_chunk = mcep[:, start:end]
            mcep_tensor = torch.tensor(mcep_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if source == self.source and target == self.target:
                    fake_mcep_chunk = self.generator_xy(mcep_tensor)
                elif source == self.target and target == self.source:
                    fake_mcep_chunk = self.generator_yx(mcep_tensor)
                else:
                    raise ValueError("Invalid pair, current model is not for the selected source and target")
                if source == self.source:
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.train_dataset.denormalizeMcep(fake_mcep_chunk, True)
                elif source == self.target:
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.train_dataset.denormalizeMcep(fake_mcep_chunk, False)
                fake_mcep_chunks.append(fake_mcep_chunk)
        fake_mcep = np.concatenate(fake_mcep_chunks, axis=1)
        fake_mcep = fake_mcep.T
        mcep_zoom_factor = (tf / fake_mcep.shape[0], 1)
        fake_mcep = zoom(fake_mcep, mcep_zoom_factor, order=1).astype(np.float64)
        synthesized_wav = u.reassembleWav(f0_converted, fake_mcep, ap, 22050, 5)
        u.saveWav(synthesized_wav, "out_synthesized.wav", 22050)

        # plt.plot(f0_target)
        # plt.title('Pitch Contour (f0)')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Pitch (Hz)')
        # plt.grid(True)
        # plt.show()

        plt.imshow(fake_mcep.T, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        plt.colorbar()
        plt.title('Fake Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCEP Coefficients')
        plt.show()

        plt.imshow(self.train_dataset.denormalizeMcep(mcep, True), aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        plt.colorbar()
        plt.title('Original Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCEP Coefficients')
        plt.show()

        # plt.imshow(ap, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        # plt.colorbar()
        # plt.title('Aperiodicity')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Value')
        # plt.show()