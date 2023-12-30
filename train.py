import torch
import torch.nn as nn
import torch.nn.functional as F

def train(net, normal_loader, abnormal_loader, optimizer, criterion, step=None):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0).float()
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    loss, losses = criterion(predict, _label, step)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    optimizer.step()

    return losses