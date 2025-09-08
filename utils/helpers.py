import yaml
import torch
from pathlib import Path
import json
from datetime import datetime



def load_config(config_path):
    """Carrega arquivo de configuração YAML"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Carrega configuração base se especificado
    if 'base' in config:
        base_config = load_config(Path(config_path).parent / config['base'])
        base_config.update(config)
        config = base_config
    
    return config

def setup_device(config):
    """Configura dispositivo (CPU/GPU)"""
    device = config.get('device', None)
    
    if device is None:
        # Decide automaticamente
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)

def set_seed(seed):
    """Define seed para reprodutibilidade"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_experiment_name(config):
    """Cria nome único para o experimento baseado na config"""
    model_name = config['model']['name']
    growth_rate = config['model']['growth_rate']
#    compression = config['model'].get('compression', 0.5)
    lr = config['training']['learning_rate']
    batch_size = config['data']['batch_size']
    epochs = config['training']['epochs']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{model_name}_gr{growth_rate}_ep{epochs}_lr{lr}_bs{batch_size}_{timestamp}"

def setup_checkpoint_dir(config):
    """Cria diretório único para checkpoints"""
    base_dir = Path(config['checkpoint'].get('base_dir', './checkpoints'))
    exp_name = create_experiment_name(config)
    checkpoint_dir = base_dir / exp_name
    
    # Criar diretório
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar config neste diretório
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return checkpoint_dir

def setup_results_dir(config,train_config):
    """Cria diretório único para checkpoints"""
    base_dir = Path(config['evaluation'].get('output_dir', './results'))
    exp_name = create_experiment_name(train_config)
    result_dir = base_dir / exp_name
    
    # Criar diretório
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar config neste diretório
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return result_dir

class TrainingLogger:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.metrics_file = self.experiment_dir / 'training_metrics.json'
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        print(f"✅ TrainingLogger criado. Arquivo: {self.metrics_file}")  # ← DEBUG
        
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """Registra métricas de uma época"""
#        print(f"🔍 log_epoch chamado! Epoch: {epoch}")  # ← DEBUG
        
        try:
            self.metrics['epochs'].append(epoch)
            self.metrics['train_loss'].append(float(train_loss))
            self.metrics['val_loss'].append(float(val_loss))
            self.metrics['train_acc'].append(float(train_acc))
            self.metrics['val_acc'].append(float(val_acc))
            self.metrics['learning_rate'].append(float(lr))
            
#            print(f"📊 Métricas adicionadas: {dict((k, v[-1]) for k, v in self.metrics.items() if v)}")  # ← DEBUG
            
            # Salva após cada época
            self.save_metrics()
            
        except Exception as e:
            print(f"❌ ERRO em log_epoch: {e}")
            import traceback
            traceback.print_exc()

    def save_metrics(self):
        """Salva métricas em JSON"""
        try:
#            print(f"💾 Salvando em: {self.metrics_file}")  # ← DEBUG
#            print(f"📦 Dados para salvar: {self.metrics}")  # ← DEBUG
            
            # Garante que o diretório existe
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
#            print(f"✅ JSON salvo com sucesso!")  # ← DEBUG
#            print(f"📏 Tamanho do arquivo: {self.metrics_file.stat().st_size} bytes")  # ← DEBUG
            
        except Exception as e:
            print(f"❌ ERRO ao salvar JSON: {e}")
            import traceback
            traceback.print_exc()

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
