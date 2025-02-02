import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as skmetrics


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = ["accuracy","precision","recall","F1-score"]

    @torch.no_grad()
    def forward(self, output, label):
        output = torch.nn.functional.softmax(output, dim=-1)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        label = np.argmax(label, axis=1)
        acc = skmetrics.accuracy_score(label, output)
        avg_f1 = skmetrics.f1_score(label, output, average="macro")
        avg_pre = skmetrics.precision_score(label, output, average="macro")
        avg_rec = skmetrics.recall_score(label, output, average="macro")
        return round(acc,4),round(avg_f1,4),round(avg_pre,4),round(avg_rec,4)

# 计算准确率
def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    avg_f1 = skmetrics.f1_score(label, output, average="macro")
    avg_pre = skmetrics.precision_score(label, output, average="macro")
    avg_rec = skmetrics.recall_score(label, output, average="macro")
    return acc,avg_f1,avg_pre,avg_rec

