import torch
import torch.nn as nn

class CorrLoss (nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, cor):
        cri = torch.nn.MSELoss()

        mask1 = cor[:] < cor.mean()
        cor1 = cor * mask1

        mask2 = cor[:] >= cor.mean() + cor.std()*2
        cor2 = cor * mask2

        return 1 * cri(cor1, torch.ones(cor1.shape).cuda() \
                + 7 * cri(cor2, torch.zeros(cor2.shape).cuda())) / 8






