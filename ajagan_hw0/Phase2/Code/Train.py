#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Network.Network import get_model, loss_fn, accuracy
from Misc.MiscUtils import (tic, toc, FindLatestModel, count_parameters, 
                            plot_metrics, plot_confusion_matrix)
from Misc.DataUtils import get_data_loaders

sys.dont_write_bytecode = True

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_CONFIGS = {
    'basic': {
        'section': 'Section 3.3 Basic CNN',
        'use_augmentation': False,
        'use_scheduler': False,
        'use_early_stopping': False,
        'checkpoint_path': '../Checkpoints/',
        'logs_path': './Logs/',
        'results_path': './Results/'
    },
    'improved': {
        'section': 'Section 3.4 Improved CNN',
        'use_augmentation': True,
        'use_scheduler': True,
        'use_early_stopping': True,
        'checkpoint_path': '../Checkpoints_Improved/',
        'logs_path': './Logs_Improved/',
        'results_path': './Results/Section3_4/'
    },
    'resnet': {
        'section': 'Section 3.5 ResNet',
        'use_augmentation': True,
        'use_scheduler': True,
        'use_early_stopping': True,
        'checkpoint_path': '../Checkpoints_ResNet/',
        'logs_path': './Logs_ResNet/',
        'results_path': './Results/Section3_5_ResNet/'
    },
    'resnext': {
        'section': 'Section 3.5 ResNeXt',
        'use_augmentation': True,
        'use_scheduler': True,
        'use_early_stopping': True,
        'checkpoint_path': '../Checkpoints_ResNeXt/',
        'logs_path': './Logs_ResNeXt/',
        'results_path': './Results/Section3_5_ResNeXt/'
    },
    'densenet': {
        'section': 'Section 3.5 DenseNet',
        'use_augmentation': True,
        'use_scheduler': True,
        'use_early_stopping': True,
        'checkpoint_path': '../Checkpoints_DenseNet/',
        'logs_path': './Logs_DenseNet/',
        'results_path': './Results/Section3_5_DenseNet/'
    }
}


class EarlyStopping:
    
    def __init__(self, patience=5, min_delta=0, path='best_model.ckpt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, test_loss, model, optimizer, epoch, train_loss, train_acc, test_acc):
        if self.best_loss is None:
            self.best_loss = test_loss
            self.save_checkpoint(model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc)
        elif test_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = test_loss
            self.best_epoch = epoch
            self.save_checkpoint(model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }, self.path)
        print(f'  Best model saved! (Test Loss: {test_loss:.4f})')


def evaluate_full_dataset(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            _, predicted = torch.max(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100.0 * correct / total
    avg_loss = total_loss / num_batches
    return acc, avg_loss, np.array(all_preds), np.array(all_labels)


def get_model_description(model_type):
    descriptions = {
        'basic': [
            'Conv2d(3, 32, 3) -> ReLU -> MaxPool(2)',
            'Conv2d(32, 64, 3) -> ReLU -> MaxPool(2)',
            'Conv2d(64, 128, 3) -> ReLU -> MaxPool(2)',
            'Flatten -> Linear(2048, 256) -> ReLU -> Linear(256, 10)'
        ],
        'improved': [
            'Conv2d(3, 64, 3) -> BatchNorm -> ReLU -> MaxPool(2)',
            'Conv2d(64, 128, 3) -> BatchNorm -> ReLU -> MaxPool(2)',
            'Conv2d(128, 256, 3) -> BatchNorm -> ReLU -> MaxPool(2)',
            'Flatten -> Linear(4096, 512) -> BatchNorm -> ReLU -> Dropout(0.5)',
            'Linear(512, 10)'
        ],
        'resnet': [
            'ResNet-18 adapted for CIFAR-10 (32x32 images)',
            'Initial: Conv2d(3, 64, 3) -> BatchNorm -> ReLU',
            'Stage 1: 2 BasicBlocks (64 channels)',
            'Stage 2: 2 BasicBlocks (128 channels, stride 2)',
            'Stage 3: 2 BasicBlocks (256 channels, stride 2)',
            'Stage 4: 2 BasicBlocks (512 channels, stride 2)',
            'Global AvgPool -> Linear(512, 10)',
            'Skip connections in each BasicBlock'
        ],
        'resnext': [
            'ResNeXt-29 (8x4d) adapted for CIFAR-10',
            'Initial: Conv2d(3, 64, 3) -> BatchNorm -> ReLU',
            'Stage 1: 3 ResNeXtBlocks (cardinality=8, width=4)',
            'Stage 2: 3 ResNeXtBlocks (cardinality=8, width=8, stride 2)',
            'Stage 3: 3 ResNeXtBlocks (cardinality=8, width=16, stride 2)',
            'Global AvgPool -> Linear(FC, 10)',
            'Grouped convolutions with cardinality=8'
        ],
        'densenet': [
            'DenseNet-100 (k=12) adapted for CIFAR-10',
            'Initial: Conv2d(3, 24, 3)',
            'Dense Block 1: 16 layers (growth_rate=12)',
            'Transition 1: 1x1 Conv -> AvgPool (compression=0.5)',
            'Dense Block 2: 16 layers (growth_rate=12)',
            'Transition 2: 1x1 Conv -> AvgPool (compression=0.5)',
            'Dense Block 3: 16 layers (growth_rate=12)',
            'BatchNorm -> ReLU -> Global AvgPool -> Linear(FC, 10)',
            'Dense connections: each layer receives all previous feature maps'
        ]
    }
    return descriptions.get(model_type, ['Unknown architecture'])


def PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples, NumTestSamples, 
                LearningRate, device, NumParams, ModelType, LatestFile):
    config = MODEL_CONFIGS[ModelType]
    print(f'\nTraining Configuration - {config["section"]}')
    print('-' * 50)
    print(f'Model: {ModelType}, Epochs: {NumEpochs}, Batch Size: {MiniBatchSize}')
    print(f'Train/Test Samples: {NumTrainSamples}/{NumTestSamples}')
    print(f'LR: {LearningRate}, Device: {device}, Parameters: {NumParams:,}')
    if config['use_augmentation']:
        print(f'Augmentation: HorizontalFlip, RandomCrop, Rotation')
    if config['use_scheduler']:
        print(f'Scheduler: CosineAnnealingLR')
    if config['use_early_stopping']:
        print(f'Early Stopping: patience=5')
    if LatestFile is not None:
        print(f'Resuming from: {LatestFile}')
    print('-' * 50)


def TrainOperation(NumEpochs, MiniBatchSize, BasePath, LearningRate, LoadCheckPoint, ModelType):
    
    config = MODEL_CONFIGS[ModelType]
    CheckPointPath = config['checkpoint_path']
    LogsPath = config['logs_path']
    ResultsPath = config['results_path']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    os.makedirs(CheckPointPath, exist_ok=True)
    os.makedirs(LogsPath, exist_ok=True)
    os.makedirs(ResultsPath, exist_ok=True)
    
    data_type = 'improved' if config['use_augmentation'] else 'basic'
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        base_path=BasePath,
        batch_size=MiniBatchSize,
        num_workers=2,
        model_type=data_type
    )
    
    NumTrainSamples = len(train_dataset)
    NumTestSamples = len(test_dataset)
    
    model = get_model(ModelType, output_size=10)
    model = model.to(device)
    
    NumParams = count_parameters(model)
    
    Optimizer = optim.AdamW(model.parameters(), lr=LearningRate, weight_decay=1e-4)
    
    if config['use_scheduler']:
        scheduler = CosineAnnealingLR(Optimizer, T_max=NumEpochs, eta_min=1e-6)
    
    if config['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=5, 
            path=os.path.join(CheckPointPath, 'best_model.ckpt')
        )
    
    Writer = SummaryWriter(LogsPath)
    
    StartEpoch = 0
    LatestFile = None
    if LoadCheckPoint:
        LatestFile = FindLatestModel(CheckPointPath)
        if LatestFile is not None:
            CheckPoint = torch.load(os.path.join(CheckPointPath, LatestFile + '.ckpt'))
            model.load_state_dict(CheckPoint['model_state_dict'])
            Optimizer.load_state_dict(CheckPoint['optimizer_state_dict'])
            StartEpoch = CheckPoint['epoch'] + 1
            print(f'Loaded checkpoint: {LatestFile}, starting from epoch {StartEpoch}')
    
    if LatestFile is None:
        print('New model initialized...')
    
    PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples, NumTestSamples,
                LearningRate, device, NumParams, ModelType, LatestFile)
    
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []
    
    print('\nStarting Training...\n')
    total_start_time = tic()
    
    for epoch in range(StartEpoch, NumEpochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NumEpochs}', leave=True)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            Optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            Optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            num_batches += 1
            
            running_loss = epoch_loss / num_batches
            running_acc = 100.0 * epoch_correct / epoch_total
            pbar.set_postfix({'loss': f'{running_loss:.2f}', 'acc': f'{running_acc:.1f}'})
        
        if config['use_scheduler']:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = LearningRate
        
        train_acc, train_loss, _, _ = evaluate_full_dataset(model, train_loader, device)
        test_acc, test_loss, _, _ = evaluate_full_dataset(model, test_loader, device)
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        
        lr_info = f', LR: {current_lr:.6f}' if config['use_scheduler'] else ''
        print(f'Epoch [{epoch+1}/{NumEpochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%{lr_info}')
        
        Writer.add_scalar('Loss/train', train_loss, epoch + 1)
        Writer.add_scalar('Loss/test', test_loss, epoch + 1)
        Writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
        Writer.add_scalar('Accuracy/test', test_acc, epoch + 1)
        if config['use_scheduler']:
            Writer.add_scalar('LearningRate', current_lr, epoch + 1)
        Writer.flush()
        
        SaveName = os.path.join(CheckPointPath, f'{epoch}model.ckpt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }, SaveName)
        
        if config['use_early_stopping']:
            if early_stopping(test_loss, model, Optimizer, epoch, train_loss, train_acc, test_acc):
                print(f'\nEarly stopping triggered at epoch {epoch+1}!')
                print(f'Best model was at epoch {early_stopping.best_epoch+1} with Test Loss: {early_stopping.best_loss:.4f}')
                break
    
    total_time = toc(total_start_time)
    print(f'\nTraining Complete! Total time: {total_time/60:.2f} minutes')
    
    if config['use_early_stopping']:
        best_model_path = os.path.join(CheckPointPath, 'best_model.ckpt')
        if os.path.exists(best_model_path):
            print(f'\nLoading best model from epoch {early_stopping.best_epoch+1} for final evaluation...')
            CheckPoint = torch.load(best_model_path)
            model.load_state_dict(CheckPoint['model_state_dict'])
    
    print('\nGenerating final confusion matrices...')
    
    train_acc_final, train_loss_final, train_preds, train_labels = evaluate_full_dataset(
        model, train_loader, device
    )
    train_cm = confusion_matrix(train_labels, train_preds)
    plot_confusion_matrix(train_cm, CIFAR10_CLASSES, 
                         os.path.join(ResultsPath, 'confusion_matrix_train.png'),
                         'Confusion Matrix - Training Data')
    
    test_acc_final, test_loss_final, test_preds, test_labels = evaluate_full_dataset(
        model, test_loader, device
    )
    test_cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(test_cm, CIFAR10_CLASSES,
                         os.path.join(ResultsPath, 'confusion_matrix_test.png'),
                         'Confusion Matrix - Test Data')
    
    actual_epochs = len(train_acc_history)
    plot_metrics(train_acc_history, test_acc_history, train_loss_history, 
                 test_loss_history, actual_epochs, ResultsPath)
    
    print(f'\nFinal Results - {config["section"]}')
    print(f'Parameters: {NumParams:,}')
    print(f'Train Loss: {train_loss_final:.4f}, Test Loss: {test_loss_final:.4f}')
    print(f'Train Accuracy: {train_acc_final:.2f}%, Test Accuracy: {test_acc_final:.2f}%')
    if config['use_early_stopping']:
        print(f'Best Epoch: {early_stopping.best_epoch+1}')
    
    print('\nConfusion Matrix (Training):')
    for i in range(10):
        print(f'{train_cm[i, :]} ({CIFAR10_CLASSES[i]})')
    print(f'Train Accuracy: {train_acc_final:.2f}%')
    
    print('\nConfusion Matrix (Test):')
    for i in range(10):
        print(f'{test_cm[i, :]} ({CIFAR10_CLASSES[i]})')
    print(f'Test Accuracy: {test_acc_final:.2f}%')
    
    with open(os.path.join(ResultsPath, 'metrics.txt'), 'w') as f:
        f.write(f'{config["section"]} Results\n')
        f.write('-' * 50 + '\n\n')
        f.write('Architecture:\n')
        for line in get_model_description(ModelType):
            f.write(f'  {line}\n')
        f.write('\n')
        f.write(f'Number of Parameters: {NumParams:,}\n')
        f.write(f'Optimizer: AdamW\n')
        f.write(f'Learning Rate: {LearningRate}\n')
        f.write(f'Batch Size: {MiniBatchSize}\n')
        f.write(f'Epochs Trained: {actual_epochs}\n')
        if config['use_augmentation']:
            f.write(f'Data Augmentation: RandomHorizontalFlip, RandomCrop(32, padding=4), RandomRotation(15)\n')
            f.write(f'Standardization: [-1, 1] normalization\n')
        if config['use_scheduler']:
            f.write(f'LR Scheduler: CosineAnnealingLR\n')
        if config['use_early_stopping']:
            f.write(f'Early Stopping: Patience=5, based on Test Loss\n')
            f.write(f'Best Epoch: {early_stopping.best_epoch+1}\n')
        f.write('\n')
        f.write(f'Final Train Loss: {train_loss_final:.4f}\n')
        f.write(f'Final Test Loss: {test_loss_final:.4f}\n')
        f.write(f'Final Train Accuracy: {train_acc_final:.2f}%\n')
        f.write(f'Final Test Accuracy: {test_acc_final:.2f}%\n')
        f.write('\nEpoch-wise Metrics:\n')
        f.write('-' * 70 + '\n')
        f.write(f'{"Epoch":<8}{"Train Loss":<15}{"Test Loss":<15}{"Train Acc":<15}{"Test Acc":<15}\n')
        f.write('-' * 70 + '\n')
        for i in range(len(train_acc_history)):
            f.write(f'{i+1:<8}{train_loss_history[i]:<15.4f}{test_loss_history[i]:<15.4f}{train_acc_history[i]:<15.2f}{test_acc_history[i]:<15.2f}\n')
    
    Writer.close()
    
    print(f'\nOutputs saved to {ResultsPath}')
    print(f'TensorBoard logs: {LogsPath}')
    print(f'Checkpoints: {CheckPointPath}')
    
    return model


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../CIFAR10', 
                       help='Base path to CIFAR10 data folder')
    Parser.add_argument('--NumEpochs', type=int, default=50, 
                       help='Number of Epochs to Train for')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, 
                       help='Size of the MiniBatch')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, 
                       help='Load from checkpoint? 1=Yes, 0=No')
    Parser.add_argument('--LearningRate', type=float, default=0.001, 
                       help='Learning rate')
    Parser.add_argument('--ModelType', default='basic', 
                       choices=['basic', 'improved', 'resnet', 'resnext', 'densenet'],
                       help='Model type')
    
    Args = Parser.parse_args()
    
    TrainOperation(
        NumEpochs=Args.NumEpochs,
        MiniBatchSize=Args.MiniBatchSize,
        BasePath=Args.BasePath,
        LearningRate=Args.LearningRate,
        LoadCheckPoint=Args.LoadCheckPoint,
        ModelType=Args.ModelType
    )


if __name__ == '__main__':
    main()
