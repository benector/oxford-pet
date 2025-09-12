import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from .custom_dataset import OxfordPetDataset
import numpy as np
from collections import defaultdict

def get_data_loaders(config):
    """Cria DataLoaders com validação balanceada por classes"""
    dataset = OxfordPetDataset(
        root=config['data']['dataset_path'],
        transform=get_transforms(config, is_train=True)
    )
    
    # Split estratificado (balanceado por classe)
    train_indices, val_indices = stratified_split(
        dataset, 
        val_ratio=config['data']['validation_split'],
        samples_per_class=19  # 19 imagens por classe para validação
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"📊 Dataset balanceado:")
    print(f"   Treino: {len(train_dataset)} amostras")
    print(f"   Validação: {len(val_dataset)} amostras")
    print(f"   Validação por classe: ~19 amostras")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader

def stratified_split(dataset, val_ratio=0.2, samples_per_class=None):
    """
    Split estratificado que garante balanceamento de classes
    """
    # Organiza amostras por classe
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset.samples[idx]  # Assumindo que dataset.samples existe
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for class_label, indices in class_indices.items():
        n_val = samples_per_class if samples_per_class else int(len(indices) * val_ratio)
        
        # Garante que não tira mais do que existe
        n_val = min(n_val, len(indices) - 1)  # Pelo menos 1 amostra no treino
        
        # Embaralha as amostras da classe
        np.random.shuffle(indices)
        
        # Separa validação
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    # Embaralha os índices finais
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    return train_indices, val_indices

def get_test_loader(config):
    """Cria DataLoader para teste"""
    dataset = OxfordPetDataset(
        root=config['data']['dataset_path'],
        transform=get_transforms(config, is_train=False)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,  # Teste não deve ser shuffle
        num_workers=config['data']['num_workers']
    )
    
    return test_loader

def get_transforms(config, is_train=True):    
    if is_train:
        return transforms.Compose([
            transforms.Resize((config['model']['input_size'], 
                             config['model']['input_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config['model']['input_size'], 
                             config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
