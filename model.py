import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
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
        self.criterion_adv = nn.MSELoss().to(self.device)
        self.criterion_cycle = nn.L1Loss().to(self.device)
        self.criterion_identity = nn.L1Loss().to(self.device)
        self.g_scheduler_xy = StepLR(self.g_optimizer_xy, step_size = 200000, gamma = 0.1)
        self.g_scheduler_yx = StepLR(self.g_optimizer_yx, step_size = 200000, gamma = 0.1)
        self.d_scheduler_x = StepLR(self.d_optimizer_x, step_size = 200000, gamma = 0.1)
        self.d_scheduler_y = StepLR(self.d_optimizer_y, step_size = 200000, gamma = 0.1)
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
        file_path = "saved_model_epoch_300.pth"
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
        path_train = "training_data/transformed_audio"
        path_eval = "evaluation_data/transformed_audio"
        self.train_dataset = ad.AudioDataset(path_train, source, target)
        source_mean_g, source_std_g = self.train_dataset.getSourceNorm(True)
        target_mean_g, target_std_g = self.train_dataset.getSourceNorm(False)
        self.eval_dataset = ad.AudioDataset(path_eval, source, target, source_mean_g, source_std_g, target_mean_g, target_std_g)

        source_dataset = u.MCEPDataset(self.train_dataset.source_mcep, ad.getId(source))
        target_dataset = u.MCEPDataset(self.train_dataset.target_mcep, ad.getId(target))
        eval_source_dataset = u.MCEPDataset(self.eval_dataset.source_mcep, ad.getId(source))
        eval_target_dataset = u.MCEPDataset(self.eval_dataset.target_mcep, ad.getId(target))

        self.source_loader = DataLoader(source_dataset, batch_size = 1, shuffle = True, num_workers = 1, pin_memory = True)
        self.target_loader = DataLoader(target_dataset, batch_size = 1, shuffle = True, num_workers = 1, pin_memory = True)
        self.eval_source_loader = DataLoader(eval_source_dataset, batch_size = 1, shuffle = True, num_workers = 1, pin_memory = True)
        self.eval_target_loader = DataLoader(eval_target_dataset, batch_size = 1, shuffle = True, num_workers = 1, pin_memory = True)

    def train(self, load_state = False, num_epochs = 2000):
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

                # Identity mapping: x -> x', y -> y'
                identity_x = self.generator_yx(source_mcep_batch)
                identity_y = self.generator_xy(target_mcep_batch)

                identity_mapping_loss = identity_loss_lambda * (self.criterion_identity(identity_x, source_mcep_batch) + self.criterion_identity(identity_y, target_mcep_batch))

                # Forward-inverse mapping: x -> y -> x', Inverse-forward mapping: y -> x -> y'
                fake_y = self.generator_xy(source_mcep_batch)
                cycle_x = self.generator_yx(fake_y)
                fake_x = self.generator_yx(target_mcep_batch)
                cycle_y = self.generator_xy(fake_x)

                cycle_consistency_loss = cycle_loss_lambda * (self.criterion_cycle(cycle_x, source_mcep_batch) + self.criterion_cycle(cycle_y, target_mcep_batch))

                # Adversarial loss for direct mappings (first step)
                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)
                d_adv1_xy = self.criterion_adv(d_fake_y, torch.ones(d_fake_y.shape).to(self.device))
                d_adv1_yx = self.criterion_adv(d_fake_x, torch.ones(d_fake_x.shape).to(self.device))

                first_adversarial_loss = d_adv1_xy + d_adv1_yx

                # Adversarial loss for cycle-consistent mappings (second step)
                d_real_x = self.discriminator_x(source_mcep_batch)
                d_real_y = self.discriminator_y(target_mcep_batch)
                d_cycle_x = self.discriminator_x(cycle_x)
                d_cycle_y = self.discriminator_y(cycle_y)
                d_adv2_xy = self.criterion_adv(d_cycle_x, torch.zeros(d_cycle_x.shape).to(self.device)) + (self.criterion_adv(d_real_x, torch.ones(d_real_x.shape).to(self.device)) + self.criterion_adv(d_fake_x, torch.zeros(d_fake_x.shape).to(self.device))) / 2
                d_adv2_yx = self.criterion_adv(d_cycle_y, torch.zeros(d_cycle_y.shape).to(self.device)) + (self.criterion_adv(d_real_y, torch.ones(d_real_y.shape).to(self.device)) + self.criterion_adv(d_fake_y, torch.zeros(d_fake_y.shape).to(self.device))) / 2

                second_adversarial_loss = d_adv2_xy + d_adv2_yx

                total_loss = cycle_consistency_loss + identity_mapping_loss + first_adversarial_loss + second_adversarial_loss

                self.resetGrad()
                total_loss.backward()
                self.g_optimizer_xy.step()
                self.g_optimizer_yx.step()
                self.d_optimizer_x.step()
                self.d_optimizer_y.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"Adv1 Loss: {first_adversarial_loss.item():.4f}, Adv2 Loss: {second_adversarial_loss.item():.4f}, "
                      f"Cycle Loss: {cycle_consistency_loss.item():.4f}, "
                      f"Identity Loss: {identity_mapping_loss.item():.4f}, "
                      f"Total Loss: {first_adversarial_loss.item() + second_adversarial_loss.item() + cycle_consistency_loss.item() + identity_mapping_loss.item():.4f}")

            if (epoch + 1) % 50 == 0:
                self.saveModel(epoch + 1)

            # self.evaluate()

            self.g_scheduler_xy.step()
            self.g_scheduler_yx.step()
            self.d_scheduler_x.step()
            self.d_scheduler_y.step()

    def evaluate(self):# add mcd and the other one
        self.loadModel()
        print("Evaluating model...")
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()

        cycle_loss_lambda = 10
        identity_loss_lambda = 5

        with torch.no_grad():
            target_iter = iter(self.target_loader)
            for i, source_sample in enumerate(self.source_loader):
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

                # Identity mapping: x -> x', y -> y'
                identity_x = self.generator_yx(source_mcep_batch)
                identity_y = self.generator_xy(target_mcep_batch)
                identity_loss = identity_loss_lambda * (self.criterion_identity(identity_x, source_mcep_batch) + self.criterion_identity(identity_y,
                                                                                                             target_mcep_batch))

                # Forward-inverse mapping: x -> y -> x', Inverse-forward mapping: y -> x -> y'
                fake_y = self.generator_xy(source_mcep_batch)
                cycle_x = self.generator_yx(fake_y)
                fake_x = self.generator_yx(target_mcep_batch)
                cycle_y = self.generator_xy(fake_x)
                cycle_loss = cycle_loss_lambda * (self.criterion_cycle(cycle_x, source_mcep_batch) + self.criterion_cycle(cycle_y, target_mcep_batch))

                # Adversarial loss for direct mappings (first step)
                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)
                adv_loss1_xy = self.criterion_adv(d_fake_y, torch.ones(d_fake_y.shape).to(self.device))
                adv_loss1_yx = self.criterion_adv(d_fake_x, torch.ones(d_fake_x.shape).to(self.device))
                adv_loss1 = adv_loss1_xy + adv_loss1_yx

                # Adversarial loss for cycle-consistent mappings (second step)
                d_real_x = self.discriminator_x(source_mcep_batch)
                d_real_y = self.discriminator_y(target_mcep_batch)
                d_cycle_x = self.discriminator_x(cycle_x)
                d_cycle_y = self.discriminator_y(cycle_y)
                adv_loss2_xy = self.criterion_adv(d_cycle_x, torch.zeros(d_cycle_x.shape).to(self.device)) + (self.criterion_adv(d_real_x, torch.ones(d_real_x.shape).to(self.device)) + self.criterion_adv(d_fake_x, torch.zeros(d_fake_x.shape).to(self.device))) / 2
                adv_loss2_yx = self.criterion_adv(d_cycle_y, torch.zeros(d_cycle_y.shape).to(self.device)) + (self.criterion_adv(d_real_y, torch.ones(d_real_y.shape).to(self.device)) + self.criterion_adv(d_fake_y, torch.zeros(d_fake_y.shape).to(self.device))) / 2
                adv_loss2 = adv_loss2_xy + adv_loss2_yx

                total_loss = cycle_loss.item() + identity_loss.item() + adv_loss1.item() + adv_loss2.item()

                print(f"Evaluation Results: "
                      f"Adv1 Loss: {adv_loss1.item():.4f}, "
                      f"Adv2 Loss: {adv_loss2.item():.4f}, "
                      f"Cycle Loss: {cycle_loss.item():.4f}, "
                      f"Identity Loss: {identity_loss.item():.4f}, "
                      f"Total Loss: {total_loss:.4f}")

    def voiceToTarget(self, source, target, path_to_source_data):
        self.loadModel()
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        data = np.load(path_to_source_data)
        log_f0 = np.log(data['f0'])
        mcep = data['mcep'].T
        if source == self.source:
            mcep = self.eval_dataset.normalizeMcep(mcep, True)
        elif source == self.target:
            mcep = self.eval_dataset.normalizeMcep(mcep, False)
        else:
            raise ValueError("Invalid pair, current model is not for the selected source and target")
        ap = data['source_parameter']
        pitch_dataset = ad.PitchDataset("training_data/transformed_audio", source, target)
        f0_converted = pitch_dataset.pitchConversion(log_f0)

        chunk_size = 128
        num_chunks = (mcep.shape[1]) // chunk_size
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
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk, True)
                elif source == self.target:
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk, False)
                fake_mcep_chunks.append(fake_mcep_chunk)

        left_frames = mcep.shape[1] - chunk_size * num_chunks
        padded_chunk = np.pad(mcep[:, mcep.shape[1] - left_frames:], ((0, 0), (0, 128 - left_frames)), mode='constant')
        padded_tensor = torch.tensor(padded_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if source == self.source and target == self.target:
                fake_mcep_chunk = self.generator_xy(padded_tensor)
            elif source == self.target and target == self.source:
                fake_mcep_chunk = self.generator_yx(padded_tensor)
            else:
                raise ValueError("Invalid pair, current model is not for the selected source and target")
            if source == self.source:
                fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk, True)
            elif source == self.target:
                fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk, False)
            fake_mcep_chunk = fake_mcep_chunk[:, 0:left_frames]
            fake_mcep_chunks.append(fake_mcep_chunk)

        fake_mcep = np.concatenate(fake_mcep_chunks, axis=1)
        fake_mcep = np.ascontiguousarray(fake_mcep.T.astype(np.float64))
        synthesized_wav = u.reassembleWav(f0_converted, fake_mcep, ap, 22050, 5)
        u.saveWav(synthesized_wav, "out_synthesized.wav", 22050)

        # plt.plot(f0_converted)
        # plt.title('Pitch Contour (f0)')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Pitch (Hz)')
        # plt.grid(True)
        # plt.show()
        #
        # plt.plot(np.exp(log_f0))
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

        if source == self.source:
            plt.imshow(self.eval_dataset.denormalizeMcep(mcep, True), aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        else:
            plt.imshow(self.eval_dataset.denormalizeMcep(mcep, False), aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
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