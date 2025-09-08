import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_optimizer(self, parameters):
        """Retorna otimizador baseado na configuração"""
        training_cfg = self.config['training']
        
        if training_cfg['optimizer'].lower() == 'adam':
            return torch.optim.Adam(
                parameters,
                lr=training_cfg['learning_rate'],
                weight_decay=training_cfg['weight_decay']
            )
        elif training_cfg['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(
                parameters,
                lr=training_cfg['learning_rate'],
                momentum=training_cfg['momentum'],
                weight_decay=training_cfg['weight_decay']
            )
        else:
            raise ValueError(f"Otimizador {training_cfg['optimizer']} não suportado")
    
    def get_scheduler(self, optimizer):
        """Retorna scheduler baseado na configuração"""
        training_cfg = self.config['training']
        
        if training_cfg['scheduler'].lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_cfg['step_size'],
                gamma=training_cfg['gamma']
            )
        elif training_cfg['scheduler'].lower() == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            return None
