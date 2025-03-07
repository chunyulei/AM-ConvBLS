import os
import shutil
import torch

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def normalize(data, dim, unbiased=True, epsilon=0.00015):
    """
    normalize dataset(preprocess)
    :param data: (batch_size, channels, width, height)
    :param dim:  dimensionality to be normalize
    :param unbiased: unbiased estimated
    :param epsilon:
    :return:
    """
    mean_ = torch.mean(data, dim=dim, keepdim=True)
    std_ = torch.sqrt(
        torch.var(data, dim=dim, unbiased=unbiased, keepdim=True) + epsilon)
    data = (data - mean_) / std_
    return data


class ZCA_parallel(object):
    def __init__(self, regularization=0.1):
        self.regularization = regularization
        self.mean_ = None
        self.components_ = None

    def fit(self, data):
        # data.shape: (batch_size, groups, channels_per_group, height, width)
        n_samples, groups, channels_per_group, height, width = data.shape
        data = data.view(n_samples, groups, -1)

        self.mean_ = torch.mean(data, dim=0)
        data = data - self.mean_
        sigma = data.permute(1, 0, 2).transpose(
            2, 1) @ data.permute(1, 0, 2) / (n_samples - 1)
        U, S, V = torch.svd(sigma)
        for idx, t in enumerate((1. / torch.sqrt(S + self.regularization))):
            tc = torch.diag(t).unsqueeze(0)
            S_diag = tc if idx == 0 else torch.cat([S_diag, tc], dim=0)
        tmp = U @ S_diag
        self.components_ = tmp @ U.transpose(2, 1)
        return self

    def transform(self, data, parallel, index):
        if parallel:
            # data.shape: (batch_size, channels, height, width)
            n_samples, groups, channels_per_group, height, width = data.shape
            data = data.view(n_samples, groups, -1)

            data_transformed = data - self.mean_
            data_transformed = data_transformed.permute(
                1, 0, 2) @ self.components_.transpose(2, 1)

            data_transformed = data_transformed.reshape(
                groups, n_samples, channels_per_group, height, width).permute(1, 0, 2, 3, 4)
        else:
            data = data.unsqueeze(1)
            n_samples, groups, channels_per_group, height, width = data.shape
            data = data.view(n_samples, 1, -1)

            data_transformed = data - self.mean_[index].unsqueeze(0)
            data_transformed = data_transformed.permute(
                1, 0, 2) @ self.components_[index].unsqueeze(0).transpose(2, 1)

            data_transformed = data_transformed.reshape(
                groups, n_samples, channels_per_group, height, width).permute(1, 0, 2, 3, 4)
        return data_transformed

    def fit_transform(self, data):
        self.fit(data=data)
        return self.transform(data=data, parallel=True, index=0)



class Accuracy(object):

    def __init__(self):
        super(Accuracy, self).__init__()
        self.reset()

    def reset(self):
        self.acc = 0
        self.correct = 0
        self.cnt = 0

    def update(self, output, target):

        assert output.shape[0] == target.shape[0], 'batch size mismatch'

        batch_size = target.shape[0]

        y_hat = output.argmax(axis=1)
        y_true = target.argmax(axis=1)

        self.correct += sum(y_hat == y_true).item()
        self.cnt += batch_size
        self.acc = round(self.correct / self.cnt, 5)


class metric(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.y_pred = list()
        self.y_true = list()

    def update(self, pred, label):
        self.y_pred += pred.cpu().numpy().tolist()
        self.y_true += label.cpu().numpy().tolist()
    
    def get_accuracy(self):
        acc = accuracy_score(
            y_true=self.y_true,
            y_pred=self.y_pred
            )
        return acc

    def get_F1_socre(self, average):
        f1 = f1_score(
            y_true=self.y_true,
            y_pred=self.y_pred,
            average=average
        )
        return f1
    
    def get_MCC(self):
        mcc = matthews_corrcoef(
            y_pred=self.y_pred,
            y_true=self.y_true
        )
        return mcc
