import torch


class MAEEmbeddingDataset(torch.utils.data.Dataset):
    """
    Loads a precomputed MaskedEmbedder embedding dump saved by the extraction script.
    """

    def __init__(self, path):
        payload = torch.load(path)
        self.embeddings = payload['embeddings']
        self.cate_idx = torch.as_tensor(payload['cate_idx'], dtype=torch.long)
        self.dataset_idx = torch.as_tensor(payload['idx'], dtype=torch.long)
        self.sid = payload.get('sid', [])
        self.mid = payload.get('mid', [])

        assert self.embeddings.ndim == 2, "Embeddings must be (N, dim)"

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        item = {
            'idx': self.dataset_idx[idx],
            'embedding': self.embeddings[idx],
            'cate_idx': self.cate_idx[idx],
            'sid': self.sid[idx] if idx < len(self.sid) else None,
            'mid': self.mid[idx] if idx < len(self.mid) else None,
        }
        return item
