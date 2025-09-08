import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils.helpers import load_config, setup_device, setup_results_dir
from data.dataset_loader import get_test_loader
from models import get_model

def test_model(config_path, train_config):
    # Carrega configuração
    config = load_config(config_path)
    train_config = load_config(train_config)
    device = setup_device(config)
    
    # Carrega dados de teste
    test_loader = get_test_loader(config)
    print(f"Dataset de teste carregado: {len(test_loader.dataset)} amostras")
    
    # Carrega modelo
    model = get_model(config).to(device)
    
    # Carrega weights do checkpoint
    checkpoint_path = config['model']['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ✅ VERIFICA TIPO DE CHECKPOINT
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint carregado: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"   Época do checkpoint: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"   Val accuracy do treino: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ State_dict carregado: {checkpoint_path}")
    
    # Teste
    model.eval()
    all_preds = []
    all_targets = []
    all_probabilities = [] 
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())
    
    # Calcula métricas
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print(f"\n📊 Resultados do Teste:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    
    #RELATÓRIO DETALHADO POR CLASSE
    print("\n📈 Relatório por classe:")
    print(classification_report(all_targets, all_preds, zero_division=0))
    
    # Salva resultados
    output_dir = Path(setup_results_dir(config,train_config))
    output_dir.mkdir(exist_ok=True)
    
    #SALVA EM JSON
    results = {
        "dataset": config['data']['dataset']['type'],
        "checkpoint_used": checkpoint_path,
        "num_samples": len(test_loader.dataset),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "per_class_metrics": classification_report(all_targets, all_preds, output_dict=True)
    }
    
    results_json = output_dir / 'test_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # SALVA RELATÓRIO EM TEXTO
    with open(output_dir / 'test_report.txt', 'w') as f:
        f.write("Test Results\n")
        f.write("============\n\n")
        f.write(f"Dataset: {config['data']['dataset']['type']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Número de amostras: {len(test_loader.dataset)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_targets, all_preds))
    
    print(f"💾 Resultados salvos em: {output_dir}/")

    with open(output_dir / "predictions.txt", "w") as f:
        # cabeçalho opcional
        f.write("target pred probs\n")
    
        for t, p, probs in zip(all_targets, all_preds, all_probabilities):
            # transforma as probabilidades em string separada por vírgula
            probs_str = ",".join([f"{prob:.4f}" for prob in probs])
            f.write(f"{t} {p} {probs_str}\n")

    print("Predições salvas em", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_config.yaml',
                       help='Caminho para arquivo de configuração')
    args = parser.parse_args()
    train_config = 'config/train_config.yaml'
    
    test_model(args.config, train_config)
