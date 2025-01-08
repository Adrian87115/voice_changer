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
        self.discriminator_x2 = d.Discriminator().to(self.device)
        self.discriminator_y2 = d.Discriminator().to(self.device)
        self.g_optimizer_xy = optim.Adam(self.generator_xy.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        self.g_optimizer_yx = optim.Adam(self.generator_yx.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        self.d_optimizer_x = optim.Adam(self.discriminator_x.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.d_optimizer_y = optim.Adam(self.discriminator_y.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.d_optimizer_x2 = optim.Adam(self.discriminator_x.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.d_optimizer_y2 = optim.Adam(self.discriminator_y.parameters(), lr = 0.0001, betas = (0.5, 0.999))
        self.criterion_adv = nn.MSELoss().to(self.device)
        self.criterion_cycle = nn.L1Loss().to(self.device)
        self.criterion_identity = nn.L1Loss().to(self.device)
        self.g_scheduler_xy = StepLR(self.g_optimizer_xy, step_size = 100000, gamma = 0.1)
        self.g_scheduler_yx = StepLR(self.g_optimizer_yx, step_size = 100000, gamma = 0.1)
        self.d_scheduler_x = StepLR(self.d_optimizer_x, step_size = 100000, gamma = 0.1)
        self.d_scheduler_y = StepLR(self.d_optimizer_y, step_size = 100000, gamma = 0.1)
        self.d_scheduler_x2 = StepLR(self.d_optimizer_x, step_size = 100000, gamma = 0.1)
        self.d_scheduler_y2 = StepLR(self.d_optimizer_y, step_size = 100000, gamma = 0.1)
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
            'discriminator_x2_state_dict': self.discriminator_x2.state_dict(),
            'discriminator_y2_state_dict': self.discriminator_y2.state_dict(),
            'g_optimizer_xy_state_dict': self.g_optimizer_xy.state_dict(),
            'g_optimizer_yx_state_dict': self.g_optimizer_yx.state_dict(),
            'd_optimizer_x_state_dict': self.d_optimizer_x.state_dict(),
            'd_optimizer_y_state_dict': self.d_optimizer_y.state_dict(),
            'd_optimizer_x2_state_dict': self.d_optimizer_x2.state_dict(),
            'd_optimizer_y2_state_dict': self.d_optimizer_y2.state_dict(),
            'iteration': self.iteration}, file_path)

    def loadModel(self):
        file_path = "saved_model_epoch_20.pth"
        checkpoint = torch.load(file_path)
        self.generator_xy.load_state_dict(checkpoint['generator_xy_state_dict'])
        self.generator_yx.load_state_dict(checkpoint['generator_yx_state_dict'])
        self.discriminator_x.load_state_dict(checkpoint['discriminator_x_state_dict'])
        self.discriminator_y.load_state_dict(checkpoint['discriminator_y_state_dict'])
        self.discriminator_x2.load_state_dict(checkpoint['discriminator_x2_state_dict'])
        self.discriminator_y2.load_state_dict(checkpoint['discriminator_y2_state_dict'])
        self.g_optimizer_xy.load_state_dict(checkpoint['g_optimizer_xy_state_dict'])
        self.g_optimizer_yx.load_state_dict(checkpoint['g_optimizer_yx_state_dict'])
        self.d_optimizer_x.load_state_dict(checkpoint['d_optimizer_x_state_dict'])
        self.d_optimizer_y.load_state_dict(checkpoint['d_optimizer_y_state_dict'])
        self.d_optimizer_x2.load_state_dict(checkpoint['d_optimizer_x2_state_dict'])
        self.d_optimizer_y2.load_state_dict(checkpoint['d_optimizer_y2_state_dict'])
        self.iteration = checkpoint['iteration']

    def resetGrad(self):
        self.g_optimizer_xy.zero_grad()
        self.g_optimizer_yx.zero_grad()
        self.d_optimizer_x.zero_grad()
        self.d_optimizer_y.zero_grad()
        self.d_optimizer_x2.zero_grad()
        self.d_optimizer_y2.zero_grad()

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

    def train(self, load_state = False, num_epochs = 1000):
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
                d2_real_x = self.discriminator_x2(source_mcep_batch)
                d2_real_y = self.discriminator_y2(target_mcep_batch)
                d2_fake_x = self.discriminator_x2(fake_x)
                d2_fake_y = self.discriminator_y2(fake_y)
                d_cycle_x = self.discriminator_x2(cycle_x)
                d_cycle_y = self.discriminator_y2(cycle_y)
                d_adv2_xy = self.criterion_adv(d_cycle_x, torch.zeros(d_cycle_x.shape).to(self.device)) + (self.criterion_adv(d2_real_x, torch.ones(d2_real_x.shape).to(self.device)) + self.criterion_adv(d2_fake_x, torch.zeros(d2_fake_x.shape).to(self.device))) / 2
                d_adv2_yx = self.criterion_adv(d_cycle_y, torch.zeros(d_cycle_y.shape).to(self.device)) + (self.criterion_adv(d2_real_y, torch.ones(d2_real_y.shape).to(self.device)) + self.criterion_adv(d2_fake_y, torch.zeros(d2_fake_y.shape).to(self.device))) / 2

                second_adversarial_loss = d_adv2_xy + d_adv2_yx

                total_loss = cycle_consistency_loss + identity_mapping_loss + first_adversarial_loss + second_adversarial_loss

                self.resetGrad()
                total_loss.backward()
                self.g_optimizer_xy.step()
                self.g_optimizer_yx.step()
                self.d_optimizer_x.step()
                self.d_optimizer_y.step()
                self.d_optimizer_x2.step()
                self.d_optimizer_y2.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(self.source_loader)}: "
                      f"Adv1 Loss: {first_adversarial_loss.item():.4f}, Adv2 Loss: {second_adversarial_loss.item():.4f}, "
                      f"Cycle Loss: {cycle_consistency_loss.item():.4f}, "
                      f"Identity Loss: {identity_mapping_loss.item():.4f}, "
                      f"Total Loss: {first_adversarial_loss.item() + second_adversarial_loss.item() + cycle_consistency_loss.item() + identity_mapping_loss.item():.4f}")

            if (epoch + 1) % 20 == 0:
                self.saveModel(epoch + 1)

            # self.evaluate()

            self.g_scheduler_xy.step()
            self.g_scheduler_yx.step()
            self.d_scheduler_x.step()
            self.d_scheduler_y.step()
            self.d_scheduler_x2.step()
            self.d_scheduler_y2.step()

    def evaluate(self):
        self.loadModel()
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        self.discriminator_x2.eval()
        self.discriminator_y2.eval()

        avg_d_loss = 0.0
        avg_g_loss = 0.0
        avg_cycle_loss = 0.0
        avg_identity_loss = 0.0
        avg_mcd = 0.0
        avg_msd = 0.0
        num_batches = 0

        with torch.no_grad():
            target_iter = iter(self.eval_target_loader)
            for i, source_sample in enumerate(self.eval_source_loader):
                try:
                    target_sample = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.eval_target_loader)
                    target_sample = next(target_iter)

                source_mcep_batch, _ = source_sample
                target_mcep_batch, _ = target_sample

                source_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in source_mcep_batch])
                target_mcep_batch = torch.stack([torch.tensor(ad.getMcepSlice(mcep)) for mcep in target_mcep_batch])

                source_mcep_batch = source_mcep_batch.transpose(-1, -2).to(self.device)
                target_mcep_batch = target_mcep_batch.transpose(-1, -2).to(self.device)

                batch_size = source_mcep_batch.size(0)
                target_batch_size = target_mcep_batch.size(0)

                if target_batch_size < batch_size:
                    repeat_factor = (batch_size + target_batch_size - 1) // target_batch_size
                    target_mcep_batch = target_mcep_batch.repeat(repeat_factor, 1, 1)[:batch_size]
                elif target_batch_size > batch_size:
                    target_mcep_batch = target_mcep_batch[:batch_size]

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
                d_target_mcep_batch = self.discriminator_y(target_mcep_batch)
                d_source_mcep_batch = self.discriminator_x(source_mcep_batch)
                d_generator_loss_xy = self.criterion_adv(d_target_mcep_batch, d_fake_y)
                d_generator_loss_yx = self.criterion_adv(d_source_mcep_batch, d_fake_x)

                # Adversarial loss for cycle-consistent mappings (second step)
                d_cycle_x = self.discriminator_x(cycle_x)
                d_cycle_y = self.discriminator_y(cycle_y)
                d_generator_loss_cycle_x = self.criterion_adv(d_source_mcep_batch, d_cycle_x)
                d_generator_loss_cycle_y = self.criterion_adv(d_target_mcep_batch, d_cycle_y)

                # Cycle and identity consistency losses
                cycle_loss = self.criterion_cycle(source_mcep_batch, cycle_x) + self.criterion_cycle(target_mcep_batch, cycle_y)
                identity_loss = self.criterion_identity(source_mcep_batch, identity_x) + self.criterion_identity(target_mcep_batch, identity_y)

                # Calculate MCD and MSD
                mcd_value = u.calculateMcd(source_mcep_batch, cycle_x)
                msd_value = u.calculateMsd(target_mcep_batch, cycle_y)

                generator_loss = 10 * cycle_loss + 5 * identity_loss
                discriminator_loss = d_generator_loss_xy + d_generator_loss_yx + d_generator_loss_cycle_x + d_generator_loss_cycle_y

                avg_d_loss += discriminator_loss.item()
                avg_g_loss += generator_loss.item()
                avg_cycle_loss += cycle_loss.item()
                avg_identity_loss += identity_loss.item()
                avg_mcd += mcd_value
                avg_msd += msd_value
                num_batches += 1

                if i == 0:
                    break

        avg_d_loss /= num_batches
        avg_g_loss /= num_batches
        avg_cycle_loss /= num_batches
        avg_identity_loss /= num_batches
        avg_mcd /= num_batches
        avg_msd /= num_batches

        print(f"Evaluation Results: D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, Cycle Loss: {avg_cycle_loss:.4f}, Identity Loss: {avg_identity_loss:.4f}, MCD: {avg_mcd:.4f}, MSD: {avg_msd:.4f}")
        return avg_d_loss, avg_g_loss, avg_cycle_loss, avg_identity_loss, avg_mcd, avg_msd

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