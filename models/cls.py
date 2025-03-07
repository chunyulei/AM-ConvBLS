import torch.nn as nn

from layers.output import Output
from layers.groutput import GROutput
from layers.daoutput import DAOutput
from layers.grdaoutput import GRDAOutput


"""output layer for ridge regression"""
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class, c):
        super(LinearClassifier, self).__init__()
        self.output = Output(in_features=feature_dim, out_features=num_class, c=c)

    def forward(self, *args):
        if self.training:
            data, label = args
            input_of_output_layer = data.double()
            out = self.output(input_of_output_layer, label)
        else:
            data = args[0]
            input_of_output_layer = data.double()
            out = self.output(input_of_output_layer)

        return out
    

"""output layer for graph regularization"""
class GRLinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class, c, lam):
        super(GRLinearClassifier, self).__init__()
        self.groutput = GROutput(in_features=feature_dim, out_features=num_class, c=c, lam=lam)

    def forward(self, *args):
        if self.training:
            data, label = args
            input_of_output_layer = data.double()
            out = self.groutput(input_of_output_layer, label)
        else:
            data = args[0]
            input_of_output_layer = data.double()
            out = self.groutput(input_of_output_layer)

        return out

"""output layer for domain adaptation"""
class DALinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class, c, lam):
        super(DALinearClassifier, self).__init__()
        self.daoutput = DAOutput(in_features=feature_dim, out_features=num_class, c=c, lam=lam)

    def forward(self, *args):
        if self.training:
            data_s, data_t, label_s = args
            data_s, data_t = data_s.double(), data_t.double()
            out = self.daoutput(data_s, data_t, label_s)
        else:
            data_t = args[0]
            data_t = data_t.double()
            out = self.daoutput(data_t)
            
        return out


"""output layer for graph regularization and domain adaptation"""
class GRDALinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class, c, lam1, lam2):
        super(GRDALinearClassifier, self).__init__()
        self.grdaoutput = GRDAOutput(in_features=feature_dim, out_features=num_class, c=c, lam1=lam1, lam2=lam2)

    def forward(self, *args):
        if self.training:
            data_s, data_t, label_s = args
            data_s, data_t = data_s.double(), data_t.double()
            out = self.grdaoutput(data_s, data_t, label_s)
        else:
            data_t = args[0]
            data_t = data_t.double()
            out = self.grdaoutput(data_t)

        return out
