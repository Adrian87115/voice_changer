import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random

import app.generator as g
import app.discriminator as d
import app.dataset as dt
import app.audio_processing as audio_p
import app.replay_buffer as rb
import app.utils as u

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

class Model():
    def __init__(self, source, target):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.app_dir)
        self.data_dir = os.path.join(self.project_root, 'data')
        self.models_dir = os.path.join(self.project_root, 'saved_models')
        self.results_dir = os.path.join(self.project_root, 'results')

        self.getData(source, target)

        self.generator_xy = g.Generator().to(self.device)
        self.generator_yx = g.Generator().to(self.device)

        self.discriminator_x = d.Discriminator().to(self.device)
        self.discriminator_y = d.Discriminator().to(self.device)
        self.discriminator_x2 = d.Discriminator().to(self.device)
        self.discriminator_y2 = d.Discriminator().to(self.device)

        self.g_optimizer = optim.Adam(itertools.chain(self.generator_xy.parameters(), 
                                                      self.generator_yx.parameters()), lr = 2e-4, betas = (0.5, 0.999))
        self.d_optimizer = optim.Adam(itertools.chain(self.discriminator_x.parameters(), 
                                                      self.discriminator_y.parameters(), 
                                                      self.discriminator_x2.parameters(), 
                                                      self.discriminator_y2.parameters()), lr = 1e-4, betas = (0.5, 0.999))
        
        self.criterion_adv = nn.MSELoss().to(self.device)
        self.criterion_cycle = nn.L1Loss().to(self.device)
        self.criterion_identity = nn.L1Loss().to(self.device)

        self.source = source
        self.target = target
        self.iteration = 0
        self.epoch_restored = 0

        self.fake_x_buffer = rb.ReplayBuffer()
        self.fake_y_buffer = rb.ReplayBuffer()
        self.cycle_x_buffer = rb.ReplayBuffer()
        self.cycle_y_buffer = rb.ReplayBuffer()

    def saveModel(self, epoch):
        file_name = f'{self.source}_{self.target}_epoch_{epoch}.pth'
        file_path = os.path.join(self.models_dir, file_name)

        torch.save({'source': self.source,
                    'target': self.target,
                    'generator_xy_state_dict': self.generator_xy.state_dict(),
                    'generator_yx_state_dict': self.generator_yx.state_dict(),

                    'discriminator_x_state_dict': self.discriminator_x.state_dict(),
                    'discriminator_y_state_dict': self.discriminator_y.state_dict(),
                    'discriminator_x2_state_dict': self.discriminator_x2.state_dict(),
                    'discriminator_y2_state_dict': self.discriminator_y2.state_dict(),

                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),

                    'iteration': self.iteration,
                    'epoch_restored': epoch}, file_path)
        
        print('\nModel saved\n')

    def loadModel(self, file_path):
        full_path = os.path.join(self.models_dir, os.path.basename(file_path))
        checkpoint = torch.load(full_path, map_location = self.device)

        source = checkpoint['source']
        target = checkpoint['target']

        assert source == self.source and target == self.target, f'Checkpoint mismatch! Expected pair: {self.source} -> {self.target}. Received: {source} -> {target}.'

        self.generator_xy.load_state_dict(checkpoint['generator_xy_state_dict'])
        self.generator_yx.load_state_dict(checkpoint['generator_yx_state_dict'])

        self.discriminator_x.load_state_dict(checkpoint['discriminator_x_state_dict'])
        self.discriminator_y.load_state_dict(checkpoint['discriminator_y_state_dict'])
        self.discriminator_x2.load_state_dict(checkpoint['discriminator_x2_state_dict'])
        self.discriminator_y2.load_state_dict(checkpoint['discriminator_y2_state_dict'])

        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        self.iteration = checkpoint['iteration']
        self.epoch_restored = checkpoint['epoch_restored']

        print('\nLoaded model\n')

    def getData(self, source, target):
        path_train = os.path.join(self.data_dir, 'training_data', 'transformed_audio')
        path_eval = os.path.join(self.data_dir, 'evaluation_data', 'transformed_audio')

        self.train_dataset = dt.AudioDataset(path_train, source, target)
        source_mean_g, source_std_g = self.train_dataset.getSourceNorm(True)
        target_mean_g, target_std_g = self.train_dataset.getSourceNorm(False)
        self.eval_dataset = dt.AudioDataset(path_eval, source, target, source_mean_g, source_std_g, target_mean_g, target_std_g)

        source_dataset = dt.MCEPDataset(self.train_dataset.source_mcep, u.getId(source))
        target_dataset = dt.MCEPDataset(self.train_dataset.target_mcep, u.getId(target))
        eval_source_dataset = dt.MCEPDataset(self.eval_dataset.source_mcep, u.getId(source))
        eval_target_dataset = dt.MCEPDataset(self.eval_dataset.target_mcep, u.getId(target))
        
        self.source_loader = DataLoader(source_dataset, batch_size = 1, shuffle = True, num_workers = 0, pin_memory = True)
        self.target_loader = DataLoader(target_dataset, batch_size = 1, shuffle = True, num_workers = 0, pin_memory = True)
        self.eval_source_loader = DataLoader(eval_source_dataset, batch_size = 1, shuffle = True, num_workers = 0, pin_memory = True)
        self.eval_target_loader = DataLoader(eval_target_dataset, batch_size = 1, shuffle = True, num_workers = 0, pin_memory = True)

        self.diagnostic_eval_source_loader = DataLoader(eval_source_dataset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)
        self.diagnostic_eval_target_loader = DataLoader(eval_target_dataset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)

    def train(self, checkpoint = None, num_save = 200, num_eval = 100, num_log = 10, num_epochs = 2470):
        if checkpoint:
            self.loadModel(checkpoint)

        print('Training model...')
        cycle_loss_lambda = 10
        identity_loss_lambda = 5

        for epoch in range(self.epoch_restored, num_epochs):
            current_epoch = epoch + 1
            target_iter = iter(self.target_loader)
            num_batches = len(self.source_loader)
            epoch_stats = {'G_Adv1': 0.0, 'G_Adv2': 0.0, 'Cycle': 0.0, 'Identity': 0.0, 'G_Total': 0.0, 'D_Adv1': 0.0, 'D_Adv2': 0.0, 'D_Total': 0.0}

            for i, source_sample in enumerate(self.source_loader):
                self.iteration += 1

                try:
                    target_sample = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_sample = next(target_iter)

                source_mcep_batch, _ = source_sample
                target_mcep_batch, _ = target_sample
                source_mcep_batch = torch.stack([torch.tensor(u.getMcepSlice(mcep)) for mcep in source_mcep_batch])
                target_mcep_batch = torch.stack([torch.tensor(u.getMcepSlice(mcep)) for mcep in target_mcep_batch])
                source_mcep_batch = source_mcep_batch.transpose(-1, -2)
                target_mcep_batch = target_mcep_batch.transpose(-1, -2)

                source_mcep_batch = source_mcep_batch.to(self.device)
                target_mcep_batch = target_mcep_batch.to(self.device)

                # ==============
                # FORWARD PASSES
                # ==============

                # Forward-inverse mapping: x -> y -> x', 
                fake_y = self.generator_xy(source_mcep_batch)
                cycle_x = self.generator_yx(fake_y)

                # Inverse-forward mapping: y -> x -> y'
                fake_x = self.generator_yx(target_mcep_batch)
                cycle_y = self.generator_xy(fake_x)

                # ================
                # GENERATOR UPDATE
                # ================

                self.g_optimizer.zero_grad()

                # Identity Loss
                if self.iteration <= 10000:
                    # Identity mapping: x -> x', y -> y'
                    identity_x = self.generator_yx(source_mcep_batch)
                    identity_y = self.generator_xy(target_mcep_batch)
                    identity_mapping_loss = identity_loss_lambda * (self.criterion_identity(identity_x, source_mcep_batch) + self.criterion_identity(identity_y, target_mcep_batch))
                else:
                    identity_mapping_loss = torch.tensor(0.0, device = self.device)

                # Cycle Loss
                cycle_consistency_loss = cycle_loss_lambda * (self.criterion_cycle(cycle_x, source_mcep_batch) + self.criterion_cycle(cycle_y, target_mcep_batch))

                # Adversarial loss for direct mappings (first step)
                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)

                d_adv1_xy = self.criterion_adv(d_fake_y, torch.ones_like(d_fake_y))
                d_adv1_yx = self.criterion_adv(d_fake_x, torch.ones_like(d_fake_x))
                
                g_first_adversarial_loss = d_adv1_xy + d_adv1_yx

                # Adversarial loss for cycle-consistent mappings (second step)
                d_cycle_x = self.discriminator_x2(cycle_x)
                d_cycle_y = self.discriminator_y2(cycle_y)
                d_adv2_xy = self.criterion_adv(d_cycle_x, torch.ones_like(d_cycle_x))
                d_adv2_yx = self.criterion_adv(d_cycle_y, torch.ones_like(d_cycle_y))

                g_second_adversarial_loss = d_adv2_xy + d_adv2_yx

                # Total Generator Loss
                g_total_loss = identity_mapping_loss + cycle_consistency_loss + g_first_adversarial_loss + g_second_adversarial_loss
                g_total_loss.backward()
                self.g_optimizer.step()

                # ====================
                # DISCRIMINATOR UPDATE
                # ====================

                self.d_optimizer.zero_grad()

                # Real scores
                d_real_x = self.discriminator_x(source_mcep_batch)
                d_real_y = self.discriminator_y(target_mcep_batch)
                d2_real_x = self.discriminator_x2(source_mcep_batch)
                d2_real_y = self.discriminator_y2(target_mcep_batch)

                # Fake scores (detached to prevent backpropagation into Generator)
                # ReplayBuffer to stop discriminator from solving the game too early
                fake_x_buffered = self.fake_x_buffer.push_and_pop(fake_x.detach())
                fake_y_buffered = self.fake_y_buffer.push_and_pop(fake_y.detach())

                cycle_x_buffered = self.cycle_x_buffer.push_and_pop(cycle_x.detach())
                cycle_y_buffered = self.cycle_y_buffer.push_and_pop(cycle_y.detach())

                d_fake_x_det = self.discriminator_x(fake_x_buffered)
                d_fake_y_det = self.discriminator_y(fake_y_buffered)

                d2_cycle_x_det = self.discriminator_x2(cycle_x_buffered)
                d2_cycle_y_det = self.discriminator_y2(cycle_y_buffered)

                # Discriminator adversarial loss for direct mappings (first step)
                d_loss_x = self.criterion_adv(d_real_x, torch.ones_like(d_real_x)) + self.criterion_adv(d_fake_x_det, torch.zeros_like(d_fake_x_det))
                d_loss_y = self.criterion_adv(d_real_y, torch.ones_like(d_real_y)) + self.criterion_adv(d_fake_y_det, torch.zeros_like(d_fake_y_det))
                
                d_first_adversarial_loss = d_loss_x + d_loss_y

                # Discriminator adversarial loss for cycle-consistent mappings (second step)
                d2_loss_x = self.criterion_adv(d2_real_x, torch.ones_like(d2_real_x)) + self.criterion_adv(d2_cycle_x_det, torch.zeros_like(d2_cycle_x_det))
                d2_loss_y = self.criterion_adv(d2_real_y, torch.ones_like(d2_real_y)) + self.criterion_adv(d2_cycle_y_det, torch.zeros_like(d2_cycle_y_det))

                d_second_adversarial_loss = d2_loss_x + d2_loss_y

                # Total Discriminator Loss
                d_total_loss = d_first_adversarial_loss + d_second_adversarial_loss
                d_total_loss.backward()
                self.d_optimizer.step()

                epoch_stats['G_Adv1'] += g_first_adversarial_loss.item()
                epoch_stats['G_Adv2'] += g_second_adversarial_loss.item()
                epoch_stats['Cycle'] += cycle_consistency_loss.item()
                epoch_stats['Identity'] += identity_mapping_loss.item()
                epoch_stats['G_Total'] += g_total_loss.item()
                epoch_stats['D_Adv1'] += d_first_adversarial_loss.item()
                epoch_stats['D_Adv2'] += d_second_adversarial_loss.item()
                epoch_stats['D_Total'] += d_total_loss.item()

            for key in epoch_stats:
                epoch_stats[key] /= num_batches

            if current_epoch % num_save == 0:
                self.saveModel(current_epoch)

            if current_epoch % num_log == 0:
                self.logEpochStats(current_epoch, epoch_stats, is_eval = False)
            else:
                print(f'Epoch [{current_epoch}/{num_epochs}]: '
                      f'G_Adv1: {epoch_stats['G_Adv1']:.4f}, '
                      f'G_Adv2: {epoch_stats['G_Adv2']:.4f}, '
                      f'Cycle: {epoch_stats['Cycle']:.4f}, '
                      f'Identity: {epoch_stats['Identity']:.4f}, '
                      f'G Loss: {epoch_stats['G_Total']:.4f}, '
                      f'D_Adv1: {epoch_stats['D_Adv1']:.4f}, '
                      f'D_Adv2: {epoch_stats['D_Adv2']:.4f}, '
                      f'D Loss: {epoch_stats['D_Total']:.4f}, '
                      f'Total Loss: {epoch_stats['G_Total'] + epoch_stats['D_Total']:.4f}')

            if current_epoch % num_eval == 0:
                self.evaluate(epoch_num = current_epoch)

    def evaluate(self, checkpoint = None, epoch_num = None):
        if checkpoint:
            self.loadModel(checkpoint)
            
        print('Evaluating model...')
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        self.discriminator_x2.eval()
        self.discriminator_y2.eval()

        cycle_loss_lambda = 10
        identity_loss_lambda = 5 

        num_batches = len(self.eval_source_loader)
        eval_stats = {'G_Adv1': 0.0, 'G_Adv2': 0.0, 'Cycle': 0.0, 'Identity': 0.0, 'G_Total': 0.0, 'D_Adv1': 0.0, 'D_Adv2': 0.0, 'D_Total': 0.0}

        with torch.no_grad():
            eval_target_batches = list(self.eval_target_loader)
            
            for i, source_sample in enumerate(self.eval_source_loader):
                target_sample = random.choice(eval_target_batches)  

                source_mcep_batch, _ = source_sample
                target_mcep_batch, _ = target_sample
                source_mcep_batch = torch.stack([torch.tensor(u.getMcepSlice(mcep)) for mcep in source_mcep_batch])
                target_mcep_batch = torch.stack([torch.tensor(u.getMcepSlice(mcep)) for mcep in target_mcep_batch])
                source_mcep_batch = source_mcep_batch.transpose(-1, -2)
                target_mcep_batch = target_mcep_batch.transpose(-1, -2)

                source_mcep_batch = source_mcep_batch.to(self.device)
                target_mcep_batch = target_mcep_batch.to(self.device)

                # ==============
                # FORWARD PASSES
                # ==============

                if self.iteration <= 10000:
                    identity_x = self.generator_yx(source_mcep_batch)
                    identity_y = self.generator_xy(target_mcep_batch)
                    identity_mapping_loss = identity_loss_lambda * (self.criterion_identity(identity_x, source_mcep_batch) + self.criterion_identity(identity_y, target_mcep_batch))
                else:
                    identity_mapping_loss = torch.tensor(0.0, device = self.device)

                fake_x = self.generator_yx(target_mcep_batch)
                fake_y = self.generator_xy(source_mcep_batch)

                cycle_x = self.generator_yx(fake_y)
                cycle_y = self.generator_xy(fake_x)

                # =========================
                # GENERATOR EVALUATION LOSS
                # =========================

                cycle_consistency_loss = cycle_loss_lambda * (self.criterion_cycle(cycle_x, source_mcep_batch) + self.criterion_cycle(cycle_y, target_mcep_batch))

                d_fake_x = self.discriminator_x(fake_x)
                d_fake_y = self.discriminator_y(fake_y)
                d_adv1_xy = self.criterion_adv(d_fake_y, torch.ones_like(d_fake_y))
                d_adv1_yx = self.criterion_adv(d_fake_x, torch.ones_like(d_fake_x))
                g_first_adversarial_loss = d_adv1_xy + d_adv1_yx

                d_cycle_x = self.discriminator_x2(cycle_x)
                d_cycle_y = self.discriminator_y2(cycle_y)
                d_adv2_xy = self.criterion_adv(d_cycle_x, torch.ones_like(d_cycle_x))
                d_adv2_yx = self.criterion_adv(d_cycle_y, torch.ones_like(d_cycle_y))
                g_second_adversarial_loss = d_adv2_xy + d_adv2_yx

                g_total_loss = identity_mapping_loss + cycle_consistency_loss + g_first_adversarial_loss + g_second_adversarial_loss

                # =============================
                # DISCRIMINATOR EVALUATION LOSS
                # =============================

                d_real_x = self.discriminator_x(source_mcep_batch)
                d_real_y = self.discriminator_y(target_mcep_batch)
                d2_real_x = self.discriminator_x2(source_mcep_batch)
                d2_real_y = self.discriminator_y2(target_mcep_batch)

                d_loss_x = self.criterion_adv(d_real_x, torch.ones_like(d_real_x)) + self.criterion_adv(d_fake_x, torch.zeros_like(d_fake_x))
                d_loss_y = self.criterion_adv(d_real_y, torch.ones_like(d_real_y)) + self.criterion_adv(d_fake_y, torch.zeros_like(d_fake_y))
                d_first_adversarial_loss = d_loss_x + d_loss_y

                d2_loss_x = self.criterion_adv(d2_real_x, torch.ones_like(d2_real_x)) + self.criterion_adv(d_cycle_x, torch.zeros_like(d_cycle_x))
                d2_loss_y = self.criterion_adv(d2_real_y, torch.ones_like(d2_real_y)) + self.criterion_adv(d_cycle_y, torch.zeros_like(d_cycle_y))
                d_second_adversarial_loss = d2_loss_x + d2_loss_y

                d_total_loss = d_first_adversarial_loss + d_second_adversarial_loss

                eval_stats['G_Adv1'] += g_first_adversarial_loss.item()
                eval_stats['G_Adv2'] += g_second_adversarial_loss.item()
                eval_stats['Cycle'] += cycle_consistency_loss.item()
                eval_stats['Identity'] += identity_mapping_loss.item()
                eval_stats['G_Total'] += g_total_loss.item()
                eval_stats['D_Adv1'] += d_first_adversarial_loss.item()
                eval_stats['D_Adv2'] += d_second_adversarial_loss.item()
                eval_stats['D_Total'] += d_total_loss.item()

        for key in eval_stats:
            eval_stats[key] /= num_batches

        epoch_label = epoch_num if epoch_num is not None else 'Final'
        self.logEpochStats(epoch_label, eval_stats, is_eval = True)

        self.generator_xy.train()
        self.generator_yx.train()
        self.discriminator_x.train()
        self.discriminator_y.train()
        self.discriminator_x2.train()
        self.discriminator_y2.train()

    def logEpochStats(self, epoch, stats, is_eval = False):
        mode = 'EVAL' if is_eval else 'TRAIN'
        
        if isinstance(epoch, int):
            log_str = f'[{mode}] Epoch {epoch:04d} | '
        else:
            log_str = f'[{mode}] Epoch {epoch} | '
            
        log_str += ' | '.join([f'{k}: {v:.4f}' for k, v in stats.items()])
        
        print(log_str)
        
        file_name = f'training_log_{self.source}_{self.target}.txt'
        file_path = os.path.join(self.results_dir, file_name)
        os.makedirs(self.results_dir, exist_ok = True)

        with open(file_path, 'a') as f:
            f.write(log_str + '\n')
            
        if not is_eval:
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
            
            stats_copy = stats.copy()
            stats_copy['epoch'] = epoch
            self.loss_history.append(stats_copy)
        else:
            if hasattr(self, 'loss_history') and len(self.loss_history) > 0:
                self.plotLosses()

    def plotLosses(self):
        epochs = [s['epoch'] for s in self.loss_history]
        g_losses = [s['G_Total'] for s in self.loss_history]
        d_losses = [s['D_Total'] for s in self.loss_history]

        plt.figure(figsize = (10, 5))
        plt.plot(epochs, g_losses, label = 'Generator Loss')
        plt.plot(epochs, d_losses, label = 'Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Convergence')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.results_dir, f'loss_plot_{self.source}_{self.target}.png')
        plt.savefig(plot_path)
        plt.close()

    def test(self, checkpoint = None):
        if checkpoint:
            self.loadModel(checkpoint)
            
        print('Testing model on metrics...')
        
        self.generator_xy.eval()
        self.generator_yx.eval()

        paired_loader = zip(self.diagnostic_eval_source_loader, self.diagnostic_eval_target_loader)
        num_batches = len(self.diagnostic_eval_source_loader)        
        test_stats = {'MCD_x2y': 0.0, 'MSD_x2y': 0.0, 'MCD_y2x': 0.0, 'MSD_y2x': 0.0}

        with torch.no_grad():
            for i, (source_sample, target_sample) in enumerate(paired_loader):
                
                source_mcep_batch, _ = source_sample
                target_mcep_batch, _ = target_sample
                
                source_mcep_batch = source_mcep_batch.clone().detach().float()
                target_mcep_batch = target_mcep_batch.clone().detach().float()
                
                source_mcep_batch = source_mcep_batch.transpose(-1, -2).to(self.device)
                target_mcep_batch = target_mcep_batch.transpose(-1, -2).to(self.device)

                fake_x = self.generator_yx(target_mcep_batch)
                fake_y = self.generator_xy(source_mcep_batch)

                np_fake_x = fake_x.squeeze(0).cpu().numpy().T
                np_source = source_mcep_batch.squeeze(0).cpu().numpy().T

                np_fake_y = fake_y.squeeze(0).cpu().numpy().T
                np_target = target_mcep_batch.squeeze(0).cpu().numpy().T

                np_fake_x = self.eval_dataset.denormalizeMcep(np_fake_x, source = True)
                np_source = self.eval_dataset.denormalizeMcep(np_source, source = True)

                np_fake_y = self.eval_dataset.denormalizeMcep(np_fake_y, source = False)
                np_target = self.eval_dataset.denormalizeMcep(np_target, source = False)

                mcd_x2y, msd_x2y = u.calculateMetrics(np_fake_y, np_target)
                mcd_y2x, msd_y2x = u.calculateMetrics(np_fake_x, np_source)

                test_stats['MCD_x2y'] += mcd_x2y
                test_stats['MCD_y2x'] += mcd_y2x
                test_stats['MSD_x2y'] += msd_x2y
                test_stats['MSD_y2x'] += msd_y2x

        for key in test_stats:
            test_stats[key] /= num_batches

        print(f'MCD (Converted X -> Real Y) : {test_stats['MCD_x2y']:.4f} dB')
        print(f'MCD (Converted Y -> Real X) : {test_stats['MCD_y2x']:.4f} dB')
        print('-' * 45)
        print(f'MSD (Converted X -> Real Y) : {test_stats['MSD_x2y']:.4f} dB')
        print(f'MSD (Converted Y -> Real X) : {test_stats['MSD_y2x']:.4f} dB')

        self.generator_xy.train()
        self.generator_yx.train()
        
        return test_stats

    def voiceToTarget(self, source, target, path_to_source_data, checkpoint):
        self.loadModel(checkpoint)
        self.generator_xy.eval()
        self.generator_yx.eval()
        self.discriminator_x.eval()
        self.discriminator_y.eval()
        
        if not os.path.isabs(path_to_source_data):
            path_to_source_data = os.path.join(self.project_root, path_to_source_data)

        data = np.load(path_to_source_data)
        f0 = data['f0']
        log_f0 = np.zeros_like(f0)
        log_f0[f0 > 0] = np.log(f0[f0 > 0])
        mcep = data['mcep']
        
        if source == self.source and target == self.target:
            mcep = self.eval_dataset.normalizeMcep(mcep, source = True)
        elif source == self.target and target == self.source:
            mcep = self.eval_dataset.normalizeMcep(mcep, source = False)
        else:
            raise ValueError('Invalid pair, current model is not for the selected source and target')
        
        mcep = mcep.T
        
        ap = data['source_parameter']
        pitch_data_path = os.path.join(self.data_dir, 'training_data', 'transformed_audio')
        pitch_dataset = dt.PitchDataset(pitch_data_path, source, target)
        f0_converted = pitch_dataset.pitchConversion(log_f0)
        f0_converted[f0 <= 0] = 0.0

        chunk_size = 128
        num_chunks = mcep.shape[1] // chunk_size
        fake_mcep_chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            mcep_chunk = mcep[:, start:end]
            mcep_tensor = torch.tensor(mcep_chunk, dtype = torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if source == self.source and target == self.target:
                    fake_mcep_chunk = self.generator_xy(mcep_tensor)
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk.T, source = False).T
                    
                elif source == self.target and target == self.source:
                    fake_mcep_chunk = self.generator_yx(mcep_tensor)
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk.T, source = True).T

                fake_mcep_chunks.append(fake_mcep_chunk)

        left_frames = mcep.shape[1] - (chunk_size * num_chunks)

        if left_frames > 0:
            padded_chunk = np.pad(mcep[:, -left_frames:], ((0, 0), (0, 128 - left_frames)), mode = 'constant')
            padded_tensor = torch.tensor(padded_chunk, dtype = torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if source == self.source and target == self.target:
                    fake_mcep_chunk = self.generator_xy(padded_tensor)
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk.T, source = False).T
                elif source == self.target and target == self.source:
                    fake_mcep_chunk = self.generator_yx(padded_tensor)
                    fake_mcep_chunk = fake_mcep_chunk.squeeze(0).cpu().numpy()
                    fake_mcep_chunk = self.eval_dataset.denormalizeMcep(fake_mcep_chunk.T, source = True).T

                fake_mcep_chunk = fake_mcep_chunk[:, :left_frames]
                fake_mcep_chunks.append(fake_mcep_chunk)

        fake_mcep = np.concatenate(fake_mcep_chunks, axis = 1)
        fake_mcep = np.ascontiguousarray(fake_mcep.T.astype(np.float64))
    
        synthesized_wav = audio_p.reassembleWav(f0_converted, fake_mcep, ap, 22050, 5)

        wav_name = f'converted_{source}_to_{target}.wav'
        wav_path = os.path.join(self.results_dir, wav_name)
        audio_p.saveWav(synthesized_wav, wav_path, 22050)
        print(f'\nConverted audio saved to {wav_path}\n')

        # plt.plot(f0_converted)
        # plt.title('Pitch Contour (f0)')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Pitch (Hz)')
        # plt.grid(True)
        # plt.show()

        # plt.plot(np.exp(log_f0))
        # plt.title('Pitch Contour (f0)')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Pitch (Hz)')
        # plt.grid(True)
        # plt.show()

        plt.figure('Fake')
        plt.imshow(fake_mcep.T, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        plt.colorbar()
        plt.title('Fake Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('MCEP Coefficients')

        plt.figure('Original')

        if source == self.source:
            denorm_mcep = self.eval_dataset.denormalizeMcep(mcep.T, True).T
        else:
            denorm_mcep = self.eval_dataset.denormalizeMcep(mcep.T, False).T
        
        plt.imshow(denorm_mcep, aspect = 'auto', origin = 'lower', cmap = 'viridis', interpolation = 'none')
        
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