import torch
import torch.nn as nn
import math
import torch.distributions as dist


class LANReLU(nn.Module):
    def __init__(self):
        super(LANReLU, self).__init__()
        #self.sigma = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(0.)*torch.rand(1)+torch.tensor(0.2), requires_grad=True)
        #self.sigma = nn.Parameter(torch.tensor(-2.), requires_grad=True)

    def forward(self, x, training=True):
        #D = self.sigma + torch.sqrt(self.sigma ** 2 + 1) %other tricks
        #D = torch.log(torch.tensor(1)+torch.exp(self.sigma))
        #D = self.sigma.clamp(min=1e-6)
        D = self.sigma
        normal_dist = dist.Normal(0, 1)
        output = x * normal_dist.cdf(x / D) + D * torch.exp(-0.5 * (x / D) ** 2) / torch.sqrt(2 * torch.tensor(math.pi))

        return output
