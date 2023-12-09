from models.GAU import *
from models.layers import *

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants
num_classes = 1000
time_window = 4 #time steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update


class mem_update(nn.Module):

    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window,
                         input.size()[1],
                         self.out_channels,
                         h,
                         w,
                         device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class GAC(nn.Module):
    def __init__(self, T, out_channels):
        super().__init__()
        self.TA = TA(T=T)
        self.SCA = SCA(in_planes=out_channels, kerenel_size=4)  # 34 K=8#18K=4
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        x_seq = x_seq.permute(1, 0, 2, 3, 4)
        spikes = spikes.permute(1, 0, 2, 3, 4)
        # x_seq B T C H W
        # spikes B T inplanes H W
        # x_seq_2 B T inplanes H W
        x_seq_2 = x_seq
        TA = self.TA(x_seq_2)
        SCA = self.SCA(x_seq_2)
        out = self.sigmoid(TA *SCA)
        y_seq = out * spikes
        y_seq = y_seq.permute(1, 0, 2, 3, 4)
        return y_seq


######################################################################################################################




class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.spike1 = mem_update()
        self.conv1 = Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False)
        self.bn1 = batch_norm_2d(out_channels)
        self.spike2 = mem_update()
        self.conv2 = Snn_Conv2d(out_channels,
                       out_channels * BasicBlock_18.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False)
        self.bn2 = batch_norm_2d1(out_channels * BasicBlock_18.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_18.expansion,
                           kernel_size=1,
                           stride=stride,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )

    def forward(self, x):
        identity = x
        x = self.bn1(self.conv1(self.spike1(x)))
        x = self.bn2(self.conv2(self.spike2(x)))
        return (x + self.shortcut(identity))


class ResNet_origin_18(nn.Module):
    # Channel:
    def __init__(self, block, num_block, zero_init_residual=True,num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3,
                       64 * k,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.T = time_window
        self.encoding = GAC(T=self.T,  out_channels=64)
        self.mem_update2 = mem_update()
        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)

        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc1_s = tdLayer(self.fc1)
        self._initialize_weights()
        

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.repeat(time_window, 1, 1, 1, 1)
        output = self.conv1(x)
        img = output
        output = self.mem_update2(output)
        output = self.encoding(img, output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output  = self.mem_update(output)
        x = self.avgpool(output)
        x = torch.flatten(x, 2)
        x = self.fc1_s(x)
        x = x.permute(1, 0, 2)
        return x



def resnet34():
    return ResNet_origin_18(BasicBlock_18, [3, 4, 6, 3],num_classes=num_classes)

def resnet18():
    return ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2],num_classes=num_classes)

if __name__ == '__main__':
    net = resnet34()
    print("Parameter numbers: {}".format(
        sum(p.numel() for p in net.parameters())))
