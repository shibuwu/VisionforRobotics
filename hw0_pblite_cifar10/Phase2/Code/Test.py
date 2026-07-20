#!/usr/bin/env python3

import torch
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Network.Network import get_model, loss_fn
from Misc.MiscUtils import count_parameters, plot_confusion_matrix
from Misc.DataUtils import get_data_loaders

sys.dont_write_bytecode = True

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

RESULTS_PATHS = {
    'basic': './Results/',
    'improved': './Results/Section3_4/',
    'resnet': './Results/Section3_5_ResNet/',
    'resnext': './Results/Section3_5_ResNeXt/',
    'densenet': './Results/Section3_5_DenseNet/'
}


def Accuracy(Pred, GT):
    return (np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred))


def ConfusionMatrix(LabelsTrue, LabelsPred, save_path=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true=LabelsTrue, y_pred=LabelsPred)
    
    print(f'\n{title}:')
    for i in range(10):
        print(f'{cm[i, :]} ({CIFAR10_CLASSES[i]})')
    
    acc = Accuracy(LabelsPred, LabelsTrue)
    print(f'Accuracy: {acc:.2f}%')
    
    if save_path:
        plot_confusion_matrix(cm, CIFAR10_CLASSES, save_path, title)
        print(f'Confusion matrix saved to: {save_path}')
    
    return cm, acc


def TestOperation(ModelPath, BasePath, MiniBatchSize, ResultsPath, ModelType):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(ResultsPath, exist_ok=True)
    
    model = get_model(ModelType, output_size=10)
    model = model.to(device)
    
    print(f'\nLoading model from: {ModelPath}')
    CheckPoint = torch.load(ModelPath, map_location=device)
    model.load_state_dict(CheckPoint['model_state_dict'])
    
    NumParams = count_parameters(model)
    print(f'Number of parameters: {NumParams:,}')
    
    if 'epoch' in CheckPoint:
        print(f'Model trained for {CheckPoint["epoch"] + 1} epochs')
    if 'test_acc' in CheckPoint:
        print(f'Test accuracy at checkpoint: {CheckPoint["test_acc"]:.2f}%')
    
    data_type = 'improved' if ModelType != 'basic' else 'basic'
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        base_path=BasePath,
        batch_size=MiniBatchSize,
        num_workers=2,
        model_type=data_type
    )
    
    print(f'\nNumber of training samples: {len(train_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')
    
    print('\nEvaluating on test set...')
    model.eval()
    test_preds = []
    test_labels = []
    test_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels_tensor = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels_tensor)
            
            _, predicted = torch.max(outputs, dim=1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            test_loss_total += loss.item()
            num_batches += 1
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_loss_avg = test_loss_total / num_batches
    
    test_cm, test_acc = ConfusionMatrix(
        test_labels, test_preds,
        save_path=os.path.join(ResultsPath, 'confusion_matrix_test.png'),
        title='Confusion Matrix - Test Data'
    )
    
    print('\nEvaluating on training set...')
    train_preds = []
    train_labels_all = []
    train_loss_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc='Evaluating Train'):
            images = images.to(device)
            labels_tensor = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels_tensor)
            
            _, predicted = torch.max(outputs, dim=1)
            
            train_preds.extend(predicted.cpu().numpy())
            train_labels_all.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            train_loss_total += loss.item()
            num_batches += 1
    
    train_preds = np.array(train_preds)
    train_labels_all = np.array(train_labels_all)
    train_loss_avg = train_loss_total / num_batches
    
    train_cm, train_acc = ConfusionMatrix(
        train_labels_all, train_preds,
        save_path=os.path.join(ResultsPath, 'confusion_matrix_train.png'),
        title='Confusion Matrix - Training Data'
    )
    
    pred_file = os.path.join(ResultsPath, 'PredOut.txt')
    with open(pred_file, 'w') as f:
        for pred in test_preds:
            f.write(f'{pred}\n')
    print(f'\nPredictions saved to: {pred_file}')
    
    section_names = {
        'basic': 'Section 3.3 Basic CNN',
        'improved': 'Section 3.4 Improved CNN',
        'resnet': 'Section 3.5 ResNet',
        'resnext': 'Section 3.5 ResNeXt',
        'densenet': 'Section 3.5 DenseNet'
    }
    
    print(f'\nFinal Results - {section_names[ModelType]}')
    print('-' * 45)
    print(f'Parameters: {NumParams:,}')
    print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%')
    
    print('\nPer-class Test Accuracy:')
    for i in range(10):
        class_mask = test_labels == i
        class_acc = np.sum(test_preds[class_mask] == i) / np.sum(class_mask) * 100
        print(f'  {CIFAR10_CLASSES[i]:12s}: {class_acc:.2f}%')
    
    print('\nMeasuring inference time...')
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    for _ in range(10):
        _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = (end_time - start_time) / num_runs * 1000
    print(f'Inference time per image: {inference_time:.4f} ms')
    
    return train_acc, test_acc, train_cm, test_cm


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', default='../Checkpoints/best_model.ckpt',
                       help='Path to model checkpoint')
    Parser.add_argument('--BasePath', default='../CIFAR10',
                       help='Base path to CIFAR10 data')
    Parser.add_argument('--MiniBatchSize', type=int, default=64,
                       help='Batch size for testing')
    Parser.add_argument('--ResultsPath', default=None,
                       help='Path to save results (auto-set based on ModelType if not provided)')
    Parser.add_argument('--ModelType', default='basic', 
                       choices=['basic', 'improved', 'resnet', 'resnext', 'densenet'],
                       help='Model type')
    
    Args = Parser.parse_args()
    
    ResultsPath = Args.ResultsPath if Args.ResultsPath else RESULTS_PATHS[Args.ModelType]
    
    TestOperation(
        ModelPath=Args.ModelPath,
        BasePath=Args.BasePath,
        MiniBatchSize=Args.MiniBatchSize,
        ResultsPath=ResultsPath,
        ModelType=Args.ModelType
    )


if __name__ == '__main__':
    main()
