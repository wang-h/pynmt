import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.autograd import Variable

SCALE_WEIGHT = 0.5 ** 0.5


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(input_size, 2 * input_size,
                                     kernel_size=(width, 1), stride=(1, 1),
                                     padding=(width // 2 * (1 - nopad), 0))
        self.dropout = nn.Dropout(dropout)
        self.init_params(dropout)

    def init_params(self, dropout):
        xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * F.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, input_size, num_layers, kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            map(GatedConv(input_size, kernel_width, dropout), range(num_layers))
        )

    def forward(self, x, hidden=None):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class WeightNormConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, init_scale=1.,
                 polyak_decay=0.9995):
        super(WeightNormConv2d, self).\
            __init__(in_channels, out_channels,
                     kernel_size, stride, padding,
                     dilation, groups, bias=True)

        self.V = self.weight
        self.g = nn.Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def init_params(self):
        self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.view(self.out_channels, -1)\
                .norm(2, 1).view(self.out_channels, *(
                    [1] * (len(self.kernel_size) + 1))).expand_as(self.V.data)
            x_init = F.conv2d(x, Variable(v_norm), None, self.stride,
                              self.padding, self.dilation, self.groups).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(
                self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(
                1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(
                x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
    def forward(self, x, init=False):
        if init is True:
            # out_channels, in_channels // groups, * kernel_size
            self.V.data.copy_(torch.randn(self.V.data.size()
                                          ).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.view(self.out_channels, -1)\
                .norm(2, 1).view(self.out_channels, *(
                    [1] * (len(self.kernel_size) + 1))).expand_as(self.V.data)
            x_init = F.conv2d(x, Variable(v_norm), None, self.stride,
                              self.padding, self.dilation, self.groups).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(
                self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(
                1), t_x_init.var(1).squeeze(1)
            # out_features
            scale_init = self.init_scale / \
                torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(
                1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(
                x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            v, g, b = get_vars_maybe_avg(
                self, ['V', 'g', 'b'], self.training,
                polyak_decay=self.polyak_decay)

            scalar = torch.norm(v.view(self.out_channels, -1), 2, 1)
            if len(scalar.size()) == 2:
                scalar = g / scalar.squeeze(1)
            else:
                scalar = g / scalar

            w = scalar.view(self.out_channels, *
                            ([1] * (len(v.size()) - 1))).expand_as(v) * v

            x = F.conv2d(x, w, b, self.stride,
                         self.padding, self.dilation, self.groups)
            return x
