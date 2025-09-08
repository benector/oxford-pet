import logging
import os
from pathlib import Path
from datetime import datetime
import torch

def setup_logger(config):
    """Configura o sistema de logging"""
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configuração do logger
    logger = logging.getLogger(config['project']['name'])
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # Formato
    formatter = logging.Formatter(config['logging']['format'])
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log das configurações iniciais
    logger.info("=" * 50)
    logger.info(f"Iniciando treino - {config['project']['name']} v{config['project']['version']}")
    logger.info("=" * 50)
    logger.info(f"Device: {config['device']}")
    logger.info(f"Seed: {config['seed']}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    return logger

def setup_tensorboard(config):
    """Configura TensorBoard se necessário"""
    if config['output'].get('tensorboard', False):
        from torch.utils.tensorboard import SummaryWriter
        
        log_dir = Path(config['output']['log_dir']) / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=log_dir)
        return writer
    return None

class TrainingLogger2:
    """Logger personalizado para acompanhamento de treino"""
    
    def __init__(self, config, tensorboard_writer=None):
        self.config = config
        self.logger = setup_logger(config)
        self.writer = tensorboard_writer
        self.epoch = 0
        
    def log_epoch(self, train_loss, val_loss, val_acc, lr):
        """Log de métricas por época"""
        self.logger.info(f"Epoch {self.epoch:03d} - "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.2f}% | "
                        f"LR: {lr:.6f}")
        
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.epoch)
            self.writer.add_scalar('Learning Rate', lr, self.epoch)
        
        self.epoch += 1
    
    def log_checkpoint(self, filename, val_acc):
        """Log de salvamento de checkpoint"""
        self.logger.info(f"Checkpoint salvo: {filename} (Acc: {val_acc:.2f}%)")
    
    def log_best_model(self, val_acc):
        """Log de melhor modelo"""
        self.logger.info(f"⭐ Novo melhor modelo! Accuracy: {val_acc:.2f}%")
    
    def log_training_end(self, best_acc, total_time):
        """Log de finalização do treino"""
        self.logger.info("=" * 50)
        self.logger.info("Treino finalizado!")
        self.logger.info(f"Melhor accuracy: {best_acc:.2f}%")
        self.logger.info(f"Tempo total: {total_time:.2f} segundos")
        self.logger.info("=" * 50)
        
        if self.writer:
            self.writer.close()

    def save_final_metrics(self, best_accuracy, total_time, final_lr, total_epochs):
        """Salva métricas finais junto com as por época"""
        try:
            final_metrics = {
                'best_accuracy': float(best_accuracy),
                'total_training_time': float(total_time),
                'final_learning_rate': float(final_lr),
                'total_epochs': int(total_epochs),
                'training_completed': True
            }
            
            # Adiciona às métricas existentes
            self.metrics['final_metrics'] = final_metrics
            
            print(f"💾 Salvando métricas finais...")
            self.save_metrics()
            
        except Exception as e:
            print(f"❌ Erro ao salvar métricas finais: {e}")
