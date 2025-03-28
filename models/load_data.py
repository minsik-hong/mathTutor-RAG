import csv
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    q, r, qshft, rshft, m = zip(*batch) # (q, r, qshft, rshft) shape
    q = pad_sequence(q, batch_first=True, padding_value=0)
    r = pad_sequence(r, batch_first=True, padding_value=0)
    qshft = pad_sequence(qshft, batch_first=True, padding_value=0)
    rshft = pad_sequence(rshft, batch_first=True, padding_value=0)
    m = pad_sequence(m, batch_first=True, padding_value=0)
    return q, r, qshft, rshft, m


class DKTDataset(Dataset):
    def __init__(self, csv_path, num_problems):
        self.samples = []
        self.num_problems = num_problems
        self._load_data(csv_path)

    def _load_data(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        for i in range(0, len(rows), 3):
            seq_length = len(rows[i + 1])
            if seq_length < 3:
                continue
            problem_seq = [int(pid) for pid in rows[i + 1] if pid != '']
            correct_seq = [int(c) for c in rows[i + 2] if c != '']
            if len(problem_seq) != len(correct_seq):
                continue
            self.samples.append((problem_seq, correct_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, r = self.samples[idx]
        r = [1 if c > 0 else 0 for c in r]
        q = torch.tensor(q, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.long)

        qshft = torch.roll(q, shifts=-1)
        rshft = torch.roll(r, shifts=-1)
        qshft[-1] = 0  # padding
        rshft[-1] = 0

        m = torch.ones_like(r, dtype=torch.bool)
        m[-1] = 0

        return q, r, qshft, rshft, m
