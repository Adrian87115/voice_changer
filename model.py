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
        self.generator = g.Generator().to(self.device)
        self.discriminator = d.Discriminator().to(self.device)
        self.domain_classifier = dc.DomainClassifier().to(self.device)
        self.getData()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.criterion_adv = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def getData(self):
        path_train = "training_data/resized_audio"
        path_eval = "evaluation_data/resized_audio"
        path_ref = "reference_data/resized_audio"
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
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        first_batch = next(iter(self.train_loader))
        mcc, speaker_id = first_batch
        self.eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        self.ref_loader = DataLoader(ref_dataset, batch_size=32, shuffle=False)

    def train(self, num_epochs=100):
        print("Training model...")
        for epoch in range(num_epochs):
            for i, sample in enumerate(self.train_loader):
                mcc_batch, speaker_id_batch = sample
                for j in range(mcc_batch.size(0)):
                    mcc = mcc_batch[j].unsqueeze(0).to(self.device)
                    speaker_id = speaker_id_batch[j].unsqueeze(0).to(self.device)
                    target_speaker_id = torch.randint(0, self.train_dataset.num_speakers, (1,)).to(self.device)
                    target_speaker_label = self.train_dataset.labels[target_speaker_id.item()]
                    emb = self.train_dataset.speaker_emb[target_speaker_label]
                    fake_mcc = self.generator(mcc, emb)

                    # Discriminator
                #     real_validity = self.discriminator(mcc, speaker_id_batch[j].unsqueeze(0).to(self.device))
                #     fake_validity = self.discriminator(fake_mcc.detach(), target_speaker_id)
                #     d_loss_real = self.criterion_adv(real_validity, torch.ones_like(real_validity))
                #     d_loss_fake = self.criterion_adv(fake_validity, torch.zeros_like(fake_validity))
                #     d_loss = (d_loss_real + d_loss_fake) / 2
                #
                #     self.d_optimizer.zero_grad()
                #     d_loss.backward()
                #     self.d_optimizer.step()
                #
                #     # Generator and Domain Classifier
                #     fake_validity = self.discriminator(fake_mcc, target_speaker_id)
                #     g_adv_loss = self.criterion_adv(fake_validity, torch.ones_like(fake_validity))
                #
                #     # Cycle consistency loss
                #     rec_mcc = self.generator(fake_mcc, target_speaker_id)
                #     cycle_loss = self.criterion_cycle(rec_mcc, mcc)
                #
                #     # Identity mapping loss
                #     id_mcc = self.generator(mcc, speaker_id)
                #     id_loss = self.criterion_identity(id_mcc, mcc)
                #
                #     # Domain classification loss
                #     cls_loss = self.criterion_cls(self.domain_classifier(fake_mcc), target_speaker_id)
                #
                #     g_loss = g_adv_loss + cycle_loss * 10 + id_loss * 5 + cls_loss
                #
                #     self.g_optimizer.zero_grad()
                #     self.c_optimizer.zero_grad()
                #     g_loss.backward()
                #     self.g_optimizer.step()
                #     self.c_optimizer.step()
                #
                # if i % 100 == 0:
                #     print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(self.train_loader)}: "
                #           f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

model = Model()
model.train()
