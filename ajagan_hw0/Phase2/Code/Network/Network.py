import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def loss_fn(out, labels):
    return F.cross_entropy(out, labels)


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = loss_fn(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = loss_fn(out, labels)
        acc = accuracy(out, labels)
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))


class CIFAR10Model(ImageClassificationBase):
    
    def __init__(self, InputSize=3*32*32, OutputSize=10):
        super(CIFAR10Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, OutputSize)
        
    def forward(self, xb):
        out = self.pool(F.relu(self.conv1(xb)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CIFAR10ModelImproved(ImageClassificationBase):
    
    def __init__(self, InputSize=3*32*32, OutputSize=10):
        super(CIFAR10ModelImproved, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, OutputSize)
        
    def forward(self, xb):
        out = self.pool(F.relu(self.bn1(self.conv1(xb))))
        out = self.pool(F.relu(self.bn2(self.conv2(out))))
        out = self.pool(F.relu(self.bn3(self.conv3(out))))
        out = out.view(out.size(0), -1)
        out = self.dropout(F.relu(self.bn_fc1(self.fc1(out))))
        out = self.fc2(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(ImageClassificationBase):
    
    def __init__(self, InputSize=3*32*32, OutputSize=10, num_blocks=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, OutputSize)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class ResNeXtBlock(nn.Module):
    expansion = 2
    
    def __init__(self, in_channels, cardinality, bottleneck_width, stride=1, downsample=None):
        super(ResNeXtBlock, self).__init__()
        
        group_width = cardinality * bottleneck_width
        
        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        
        self.conv3 = nn.Conv2d(group_width, group_width * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(group_width * self.expansion)
        
        self.downsample = downsample
        self.out_channels = group_width * self.expansion
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNeXt(ImageClassificationBase):
    
    def __init__(self, InputSize=3*32*32, OutputSize=10, cardinality=8, 
                 bottleneck_width=64, num_blocks=[3, 3, 3]):
        super(ResNeXt, self).__init__()
        
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, OutputSize)
        
        self._initialize_weights()
    
    def _make_layer(self, num_blocks, stride):
        downsample = None
        out_channels = self.cardinality * self.bottleneck_width * ResNeXtBlock.expansion
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResNeXtBlock(self.in_channels, self.cardinality, 
                                   self.bottleneck_width, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResNeXtBlock(self.in_channels, self.cardinality, 
                                       self.bottleneck_width))
        
        self.bottleneck_width *= 2
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class DenseLayer(nn.Module):
    
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseLayer, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 
                               kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super(DenseBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layers.append(DenseLayer(layer_in_channels, growth_rate, bn_size))
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(ImageClassificationBase):
    
    def __init__(self, InputSize=3*32*32, OutputSize=10, growth_rate=12, 
                 block_config=[16, 16, 16], compression=0.5, bn_size=4):
        super(DenseNet, self).__init__()
        
        self.growth_rate = growth_rate
        
        num_init_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        num_features = num_init_features
        
        self.dense1 = DenseBlock(num_features, growth_rate, block_config[0], bn_size)
        num_features = self.dense1.out_channels
        self.trans1 = Transition(num_features, int(num_features * compression))
        num_features = int(num_features * compression)
        
        self.dense2 = DenseBlock(num_features, growth_rate, block_config[1], bn_size)
        num_features = self.dense2.out_channels
        self.trans2 = Transition(num_features, int(num_features * compression))
        num_features = int(num_features * compression)
        
        self.dense3 = DenseBlock(num_features, growth_rate, block_config[2], bn_size)
        num_features = self.dense3.out_channels
        
        self.bn_final = nn.BatchNorm2d(num_features)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, OutputSize)
        
        self.num_features = num_features
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def get_model(model_type, input_size=3*32*32, output_size=10):
    if model_type == 'basic':
        return CIFAR10Model(input_size, output_size)
    
    elif model_type == 'improved':
        return CIFAR10ModelImproved(input_size, output_size)
    
    elif model_type == 'resnet':
        return ResNet(input_size, output_size, num_blocks=[2, 2, 2, 2])
    
    elif model_type == 'resnext':
        return ResNeXt(input_size, output_size, cardinality=8, 
                       bottleneck_width=16, num_blocks=[3, 3, 3])
    
    elif model_type == 'densenet':
        return DenseNet(input_size, output_size, growth_rate=12, 
                        block_config=[16, 16, 16], compression=0.5)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
