import torch
import torch.nn as nn
from pathlib import Path
import argparse
import time 

from utils.helpers import load_config, setup_device, set_seed, setup_checkpoint_dir, TrainingLogger

from data.dataset_loader import get_data_loaders
from models import get_model
from utils.logger import setup_logger

def train_model(config_path):

    start_time = time.time()

    # Carrega configuração
    config = load_config(config_path)
    set_seed(config['seed'])
    device = setup_device(config)
    
    # Setup logger
    logger = setup_logger(config)

    checkpoint_dir = setup_checkpoint_dir(config)
    config['checkpoint']['save_dir'] = str(checkpoint_dir)
    
    logger.info(f"📁 Checkpoint dir: {checkpoint_dir}")
    
    # Carrega dados
    train_loader, val_loader = get_data_loaders(config)
    logger.info(f"Dataset carregado: {len(train_loader.dataset)} treino, "
               f"{len(val_loader.dataset)} validação")
    
    # Carrega modelo
    model = get_model(config).to(device)
    logger.info(f"Modelo {config['model']['name']} carregado")
    
    # Otimizador e scheduler
    optimizer = model.get_optimizer(model.parameters())
    if config['training']['scheduler']:
        scheduler = model.get_scheduler(optimizer)
    else:
        scheduler = None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Treinamento
    best_acc = 0.0
    metrics_logger = TrainingLogger(checkpoint_dir)
    
    for epoch in range(config['training']['epochs']):
        # Fase de treino
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)  
            loss = criterion(output, target)  
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()  
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Fase de validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)  
                val_loss += criterion(output, target).item()  
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        #  REGISTRA MÉTRICAS NO JSON
        current_lr = optimizer.param_groups[0]['lr']
        metrics_logger.log_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=current_lr
        )
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss} Val Loss: {val_loss},"
                  f"Train acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        # Atualiza scheduler e salva checkpoint
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)   # precisa passar métricas
            else:
                scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, Path(config['checkpoint']['save_dir']) / 'best_model.pth')

        if config['checkpoint'].get('save_last', True):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, Path(config['checkpoint']['save_dir']) / 'last_epoch.pth')

        if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, Path(config['checkpoint']['save_dir']) / f'epoch_{epoch+1}.pth')
           
    total_time = time.time() - start_time
    metrics_logger.save_final_metrics(
    best_accuracy=best_acc,
    total_time= total_time,
    final_lr=current_lr,
    total_epochs=epoch + 1
)

    print(f"⏰ TEMPO TOTAL: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Caminho para arquivo de configuração')
    args = parser.parse_args()
    
    train_model(args.config)
