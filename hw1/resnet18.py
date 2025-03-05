'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            # Init weights with kaiming
            nn.init.kaiming_normal_(self.shortcut[0].weight)
        
        # Init weights with kaiming
        # https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Go through conv1, bn1, relu
        p1 = self.conv1(x)
        p1 = self.bn1(p1)
        p1 = nn.ReLU()(p1)
        # 2. Go through conv2, bn2
        p2 = self.conv2(p1)
        p2 = self.bn2(p2)
        # 3. Combine with shortcut output, and go through relu
        p3 = p2 + self.shortcut(x)
        p3 = nn.ReLU()(p3)

        return p3


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

        # Init weights with kaiming
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ input images and output logits """
        x = self.conv1(images)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=(2, 3))
        x = self.linear(x)
        return x

    def visualize(self, logdir: str) -> None:
        """ Visualize the kernel in the desired directory """
        # Standardize the weights of the conv1 layer in order to have a grayscale image
        conv1_weights = self.conv1.weight
        utils.save_image(conv1_weights, os.path.join(logdir, 'conv1.png'))

        conv1_weights = self.conv1.weight.mean(dim=1).unsqueeze(1)
        utils.save_image(conv1_weights, os.path.join(logdir, 'conv1_mean.png'))

        conv1_weights = (conv1_weights - conv1_weights.min()) / (conv1_weights.max() - conv1_weights.min())
        utils.save_image(conv1_weights, os.path.join(logdir, 'conv1_std.png'))
    
    def get_gradient_flow(self) -> dict:
        gradients = {
            'hidden_layers': [],
            'output_layer': None
        }
        
        # Gradient de la première couche de convolution
        if self.conv1.weight.grad is not None:
            grad_mean = self.conv1.weight.grad.abs().mean().item()
            gradients['hidden_layers'].append(grad_mean)
        
        # Gradients des couches de ResNet (layer1, layer2, layer3, layer4)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            layer_grads = []
            for block in layer:  # Chaque layer contient 2 BasicBlocks
                if block.conv1.weight.grad is not None and block.conv2.weight.grad is not None:
                    # Moyenne des gradients des deux convolutions du block
                    block_grad = (block.conv1.weight.grad.abs().mean().item() + 
                                block.conv2.weight.grad.abs().mean().item()) / 2
                    layer_grads.append(block_grad)
            if layer_grads:
                # Moyenne des gradients des blocks dans la layer
                gradients['hidden_layers'].append(sum(layer_grads) / len(layer_grads))
            
        # Gradient de la couche linéaire finale
        if self.linear.weight.grad is not None:
            grad_mean = self.linear.weight.grad.abs().mean().item()
            gradients['output_layer'] = grad_mean
            
        return gradients
