import torch as pt
import torch.nn as nn

from collections import OrderedDict

class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super(FeedforwardNetwork, self).__init__()
        
        # Create all feed-forward layers
        layers = [5, 100, 100, 100, 100, 1]
        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, pt.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), pt.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # Combine all layers in a single Sequential object to keep track of parameter count
        self.layers = pt.nn.Sequential(layerDict)

        # Bookkeeping
        self.n_trainable_parameters = sum(p.numel() for p in self.parameters())
        print('Number of Trainable Parameters:', self.n_trainable_parameters)

    def forward(self, x):
        return self.layers(x)