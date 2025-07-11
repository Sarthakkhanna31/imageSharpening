
import torch.nn as nn

class HookedDnCNN(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        layers = [nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers += [nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(64, channels, 3, padding=1)]
        self.dncnn = nn.Sequential(*layers)
        self.features = None
        self.dncnn[2].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        return x - self.dncnn(x)
