import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Carrega caminhos das imagens e labels"""
        samples = []
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_label = int(class_dir.name)
                for img_path in class_dir.glob('*.jpg'):
                    samples.append((img_path, class_label))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
class OxfordPetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root: caminho da pasta principal do dataset (onde ficam 'images' e 'annotations')
        split: 'train' ou 'test'
        transform: transformações de imagem (torchvision.transforms)
        """
        self.root = root
        self.transform = transform
        self.samples = []
        split_file = "trainval.txt" if split == "train" else "test.txt"
        list_file = os.path.join(root, "annotations", split_file)

        with open(list_file, "r") as f:
            lines = f.readlines()[6:]  # pula cabeçalho

            for line in lines:
                #print('line:', line)
                name, class_id, species_id, breed_id = line.strip().split()
                img_path = os.path.join(root, "images", name + ".jpg")
                if not os.path.exists(img_path):
                    print(f"Arquivo não encontrado: {img_path}")
                    continue
                self.samples.append((img_path, int(class_id) - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
