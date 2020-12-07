import torch
# from modules import InPlaceABN, InPlaceABNSync
from modules import InPlaceABN , ABN as InPlaceABNSync


from torch import nn
import torch.nn.functional as F
from losser import Loss


class BottleNeckResblock(nn.Module):

    # expansion = 2 #default = 4  but I think it should be 2......in our item...

    def __init__(self, inplanes, planes, index , senet_global_average, expansion = 2, stride=1, downsample=None):

        super(BottleNeckResblock, self).__init__()

        self.bn1 = InPlaceABNSync(inplanes)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)

        self.bn2 = InPlaceABNSync(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = InPlaceABNSync(planes)

        self.conv3 = nn.Conv3d(planes, planes * expansion, kernel_size=1, bias=False)

        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

        self.se_compont = SEComponent(planes * expansion, index , senet_global_average, se_resize_factor = 8)

    def forward(self, x):

        residual = x

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.conv3(out)


        if self.downsample is not None:

            residual = self.downsample(x)

        out = self.se_compont(out)

        out += residual

        return out

class SEComponent(nn.Module):

    def __init__(self, inplanes, index , senet_global_average, se_resize_factor = 4):

        super(SEComponent, self).__init__()

        self.se_resize_factor = se_resize_factor

        current_global_kernel_size = senet_global_average[index]

        self.global_average_pool = nn.AvgPool3d(current_global_kernel_size,stride=1)

        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(inplanes / se_resize_factor))
        self.fc2 = nn.Linear(in_features=round(inplanes / se_resize_factor), out_features=inplanes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, out):

        original_out = out
        out = self.global_average_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1, 1)
        out = out * original_out

        return out

class SENet(nn.Module):

    def __init__(self, block = BottleNeckResblock , layers =  [3,4, 23,3], num_classes=2):

        super(SENet, self).__init__()

        # block_channel_in = [32,64,128,256]
        block_channel_planes = [32, 64, 128, 256]

        senet_global_average = [48,24,12,6]

        self.inplanes = 32

        # self.preblock = nn.Sequential(nn.Conv3d(2, self.inplanes, kernel_size=3,stride=1, padding=1, bias=False),
        #                               InPlaceABNSync(self.inplanes),
        #                               nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3,stride=1, padding=1, bias=False))

        self.preblock = nn.Conv3d(2, self.inplanes, kernel_size=3,stride=1, padding=1, bias=False)

        # self.begin_con = nn.Conv3d(2, self.inplanes, kernel_size=3,stride=1, padding=1, bias=False)
        # self.begin_con_bn = InPlaceABNSync(self.inplanes)

        self.avgpool = nn.AvgPool3d(6, stride=1)

        self.finally_bn = InPlaceABNSync(block_channel_planes[-1] * 2)

        self.binary_class_layer   = nn.Linear(block_channel_planes[-1] * 2, 2)
        self.multiple_class_layer = nn.Linear(block_channel_planes[-1] * 2, 6)

        self.layer1 = self.make_res_block(block, block_channel_planes[0], layers[0], 0, senet_global_average, expansion = 2, stride=1,down_sample_padding=0)
        self.layer2 = self.make_res_block(block, block_channel_planes[1], layers[1], 1, senet_global_average, expansion = 2, stride=2)
        self.layer3 = self.make_res_block(block, block_channel_planes[2], layers[2], 2, senet_global_average, expansion = 2, stride=2)
        self.layer4 = self.make_res_block(block, block_channel_planes[3], layers[3], 3, senet_global_average, expansion = 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, InPlaceABNSync):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_res_block(self, block, planes, blocks, se_index, se_average_size, stride=1, expansion = 2, down_sample_padding = 0, can_down_sample = True):

        downsample = None

        if (stride != 1 or self.inplanes != planes * expansion) and can_down_sample == True: #需要下采样
            downsample = nn.Sequential(
                InPlaceABNSync(self.inplanes, activation='none'),
                nn.Conv3d(self.inplanes, planes * expansion, kernel_size=1, stride=stride, padding=down_sample_padding, bias=False))

        layers = []
        layers.append(block(self.inplanes, planes, se_index, se_average_size, expansion = expansion, stride = stride, downsample = downsample))
        # self.inplanes = planes * block.expansion
        self.inplanes = planes * expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, se_index, se_average_size, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.preblock(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.finally_bn(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        binary_class   = self.binary_class_layer(x)
        multiple_class = self.multiple_class_layer(x)

        return binary_class, multiple_class

def get_model(type, args):

    if type == "base":

        net = SENet()

        loss = Loss(args)

        get_pbb = None

    else:

        raise ("Type Error")

    return args, net, loss, get_pbb
