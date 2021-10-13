from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class LogRatio(nn.Module):
    def __init__(self, Omega=0.1, **kwargs):
        super(LogRatio, self).__init__()
        self.Omega = Omega

    def forward(self, inputs, labels):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = similarity(inputs)
        # print(sim_mat)
        labels = labels.cuda()
        kk = labels.size(1)
        targets = labels[:,0]
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        pos_mask = pos_mask - eyes_.eq(1)

        neg_mask = []
        neg_mask_ = []

        for i in range(kk-1,-1,-1):
            pos_mask_inter = labels[:,i].expand(n, n).eq(targets.expand(n, n).t())
            neg_mask_inter = eyes_.eq(eyes_) - pos_mask_inter
            neg_mask_.append(neg_mask_inter)
            if i == kk-1:
                neg_mask.append(neg_mask_inter)
            else:
                neg_mask.append(neg_mask_inter-neg_mask_[kk-i-2])
        loss = 0
        epsilon = 1e-6
        Omega = torch.tensor(self.Omega).float()
        margin = torch.tensor([0.4,0.7]).float()
        for j in range(n):
            for k in range(n):
                if pos_mask[j,k] == 1:
                    for l in range(n):
                        for m in range(kk):
                            if neg_mask[m][j,l] == 1:
                                loss_dist = torch.log(sim_mat[j,k]+epsilon) - torch.log(sim_mat[j,l]+epsilon)
                                loss_pair = torch.log(Omega+epsilon) - torch.log(Omega.pow(kk-m+1)+epsilon)
                                loss += (loss_dist - 0.1*loss_pair).pow(2)
        prec = 0
        mean_neg_sim = 0
        mean_pos_sim = 0
        return loss, prec, mean_pos_sim, mean_neg_sim





