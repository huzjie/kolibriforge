"""Data loading / splitting pipeline helpers.

Keeps the training loops tidy: load a synthetic dataset, split into train/eval,
and iterate in batches. Zero-dependency, deterministic ordering.
"""


class DataPipeline:
    def __init__(self, dataset, train_ratio=0.8, seed=0):
        self.dataset = list(dataset)
        self.train_ratio = train_ratio
        self.seed = seed
        n = len(self.dataset)
        cut = int(n * train_ratio)
        # deterministic order (stable, not random.shuffle which varies per run)
        order = sorted(range(n), key=lambda i: (i * 2654435761 + seed) % n)
        self.train = [self.dataset[i] for i in order[:cut]]
        self.eval = [self.dataset[i] for i in order[cut:]]

    def batches(self, split="train", batch_size=8):
        data = self.train if split == "train" else self.eval
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def __len__(self):
        return len(self.dataset)
