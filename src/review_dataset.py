from typing import List
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelBinarizer




class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], vectorizer, labels: List[str] = None):
        self.vectorizer = vectorizer
        self.texts = texts
        self.input_vects = vectorizer.vectorize_input(texts)
        # labels
        if labels is not None:
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(labels)
            self.label_vects = torch.from_numpy(self.label_binarizer.transform(labels)).long()
        else:
            self.label_vects = None


    def __getitem__(self, index):
        return (self.input_vects[index], None if self.label_vects is None else self.label_vects[index])

    def __len__(self):
        return len(self.input_vects)










