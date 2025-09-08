import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from .custom_dataset import OxfordPetDataset

def get_data_loaders(config):
    """Cria DataLoaders para treino e validação"""
    dataset = OxfordPetDataset(
        root=config['data']['dataset_path'],#caminho definido no test_config ou train_config
        transform=get_transforms(config, is_train=True)
    )
    
    # Split treino/validação
    val_size = int(len(dataset) * config['data']['validation_split'])#validação sendo 20% 
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
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

def get_test_loader(config):
    """Cria DataLoader para teste"""
    dataset = OxfordPetDataset(
        root=config['data']['dataset_path'],
        transform=get_transforms(config, is_train=False)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers']
    )
    
    return test_loader

def get_transforms(config, is_train=True):    
    if is_train:
        return transforms.Compose([
            transforms.Resize((config['model']['input_size'], 
                             config['model']['input_size'])),
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
#Opções de transformação para o treino
# transforms.RandomHorizontalFlip(),
#transforms.RandomRotation(10),
