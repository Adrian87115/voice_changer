import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import generator as g
import discriminator as d
import domain_classifier as dc
import audio_dataset as ad

class Model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_source_speakers = 8
        self.num_target_speakers = 4
        self.getData()
        self.generator = g.Generator(self.train_dataset.num_speakers).to(self.device)
        self.discriminator = d.Discriminator().to(self.device)
        self.domain_classifier = dc.DomainClassifier(self.num_target_speakers).to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.criterion_adv = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()

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

    def train(self, num_epochs=10):
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

                    # Update generator and domain classifier
                    self.g_optimizer.zero_grad()
                    self.c_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    self.c_optimizer.step()

                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(self.source_loader)}: "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, CLS Loss: {cls_loss.item():.4f}")

            print(f"Epoch [{epoch}/{num_epochs}] completed.")


model = Model()
model.train()
