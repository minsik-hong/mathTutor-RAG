import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from load_data import DKTDataset, collate_fn
from dkt_plus import DKTPlus


class DKTTrainer:
    def __init__(self, train_path, valid_path, num_problems, save_path='./results/model_best.pth',
                 batch_size=32, num_epochs=25, learning_rate=0.0041, hidden_size=102,
                 emb_size=102, lambda_r=0.1, lambda_w1=0.03, lambda_w2=3.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_problems = num_problems
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.save_path = save_path

        # Dataset & DataLoader
        self.train_dataset = DKTDataset(train_path, num_problems)
        self.valid_dataset = DKTDataset(valid_path, num_problems)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Model & Optimizer
        self.model = DKTPlus(num_q=num_problems, emb_size=emb_size, hidden_size=hidden_size,
                             lambda_r=lambda_r, lambda_w1=lambda_w1, lambda_w2=lambda_w2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_auc = 0

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            losses = []

            for q, r, qshft, rshft, m in self.train_loader:
                q, r, qshft, rshft, m = q.long().to(self.device), r.float().to(self.device), qshft.long().to(self.device), rshft.float().to(self.device), m.to(self.device)

                y = self.model(q, r)
                y_curr = (y * torch.nn.functional.one_hot(q, num_classes=self.num_problems).float()).sum(-1)
                y_next = (y * torch.nn.functional.one_hot(qshft, num_classes=self.num_problems).float()).sum(-1)

                y_curr = torch.masked_select(y_curr, m)
                y_next = torch.masked_select(y_next, m)
                r = torch.masked_select(r, m)
                rshft = torch.masked_select(rshft, m)

                loss_w1 = torch.masked_select(torch.norm(y[:, 1:] - y[:, :-1], p=1, dim=-1), m[:, 1:])
                loss_w2 = torch.masked_select(torch.norm(y[:, 1:] - y[:, :-1], p=2, dim=-1) ** 2, m[:, 1:])

                loss = torch.nn.functional.binary_cross_entropy(y_next, rshft) + \
                       self.model.lambda_r * torch.nn.functional.binary_cross_entropy(y_curr, r) + \
                       self.model.lambda_w1 * loss_w1.mean() / self.num_problems + \
                       self.model.lambda_w2 * loss_w2.mean() / self.num_problems

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                losses.append(loss.item())

            # Validation
            auc = self.evaluate()
            print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f} | Valid AUC: {auc:.4f}")

            if auc > self.max_auc:
                torch.save(self.model.state_dict(), self.save_path)
                self.max_auc = auc
                print(f"Model saved. Best AUC: {self.max_auc:.4f}")

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for q, r, qshft, rshft, m in self.valid_loader:
                q, r, qshft, rshft, m = q.to(self.device), r.to(self.device), qshft.to(self.device), rshft.to(self.device), m.to(self.device)
                y = self.model(q, r)
                y_next = (y * torch.nn.functional.one_hot(qshft, num_classes=self.num_problems).float()).sum(-1)
                y_next = torch.masked_select(y_next, m).cpu().numpy()
                rshft = torch.masked_select(rshft, m).cpu().numpy()
                all_preds.extend(y_next)
                all_labels.extend(rshft)

        auc = metrics.roc_auc_score(all_labels, all_preds)
        return auc


if __name__ == '__main__':
    trainer = DKTTrainer(
        train_path='/home/hail/knowledge-tracing-collection-pytorch/data/i-scream/i-scream_train.csv',
        valid_path='/home/hail/knowledge-tracing-collection-pytorch/data/i-scream/i-scream_valid.csv',
        num_problems=1865,
        batch_size=32,
        num_epochs=25,
        learning_rate=0.004155923499457689,
        hidden_size=102,
        emb_size=102,
        lambda_r=0.1,
        lambda_w1=0.03,
        lambda_w2=3.0
    )
    trainer.train()
