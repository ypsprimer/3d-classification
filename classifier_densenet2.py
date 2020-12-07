import torch
from modules import InPlaceABNSync
# from modules import InPlaceABN , InPlaceABNSync
# from modules import InPlaceABN as InPlaceABNSync


from torch import nn
import torch.nn.functional as F
from losser import Loss
import math

class TransitionLayer(nn.Module):

    def __init__(self, input_channel_number, output_channel_number):

        super(TransitionLayer, self).__init__()

        self.bn   = InPlaceABNSync(input_channel_number)
        self.conv = nn.Conv3d(input_channel_number, output_channel_number, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):

        current = self.bn(x)
        current = self.conv(current)
        current = self.pool(current)

        return current


class BachActivateConvLayer(nn.Module):

    def __init__(self, channel_in, growth_rate, bottleneck_size_basic_factor ,drop_ratio=0.8):

        super(BachActivateConvLayer, self).__init__()

        self.drop_ratio = drop_ratio
        self.growth_rate = growth_rate
        self.bottleneck_channel_out = bottleneck_size_basic_factor * growth_rate

        self.mode_bn    = InPlaceABNSync(channel_in)
        self.mode_conv  = nn.Conv3d(channel_in, self.bottleneck_channel_out, kernel_size=1, stride=1, bias=False)

        self.bn = InPlaceABNSync(self.bottleneck_channel_out)
        self.conv = nn.Conv3d(self.bottleneck_channel_out, growth_rate, kernel_size=3, stride=1,padding=1,bias=False)

        self.drop_out = nn.Dropout3d(p=self.drop_ratio)


    def forward(self, x):

        current = x
        current = self.mode_bn(current)
        current = self.mode_conv(current)

        current = self.bn(current)
        current = self.conv(current)

        if self.drop_ratio > 0:

            current = self.drop_out(current)

        return current

class DenseBlock(nn.Module):

    def __init__(self, current_block_layers_number, channel_in, growth_rate, bottleneck_size_basic_factor, drop_ratio=0.8):

        super(DenseBlock, self).__init__()

        self.channel_in = channel_in
        self.growth_rate     = growth_rate
        self.bottleneck_size_basic_factor = bottleneck_size_basic_factor
        self.current_channel_in = self.channel_in
        self.current_blcok_drop_ratio = drop_ratio
        self.current_block_layer_number = current_block_layers_number

        for i in range(self.current_block_layer_number):


            current_block_layers = BachActivateConvLayer(self.current_channel_in, self.growth_rate, self.bottleneck_size_basic_factor, self.current_blcok_drop_ratio)

            setattr(self, 'block_layer_' + str(i), current_block_layers)

            self.current_channel_in += self.growth_rate

    def get_current_block_channel_out(self):

        return self.current_channel_in


    def forward(self, x):

        current = x

        for i in range(self.current_block_layer_number):

            tmp = getattr(self, 'block_layer_' + str(i))(current)

            current = torch.cat((current, tmp), 1)

        return current

class DenseNet(nn.Module):

    # def __init__(self, growth_rate=16, block_config=(8, 8, 8, 8), compression=0.5, num_init_features=24, bottleneck_size_basic_factor=4, drop_rate=0, num_classes=2, small_inputs=True):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4), compression=0.5, num_init_features=24, bottleneck_size_basic_factor=4, drop_rate=0, num_classes=2, small_inputs=True):

        super(DenseNet, self).__init__()

        self.features = nn.Conv3d(2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_feature_channel_number = num_init_features
        self.growth_rate = growth_rate
        self.compression = compression
        self.number_class = num_classes
        self.block_config = block_config

        num_features = num_init_features


        self.dense_trainsition_out_put_list = []

        for i, num_layers in enumerate(self.block_config):

            block = DenseBlock(num_layers, num_features, self.growth_rate, bottleneck_size_basic_factor, drop_rate)

            setattr(self, 'block_' + str(i), block)

            num_features = num_features + num_layers * growth_rate


            if i != len(block_config) - 1:

                transition_layer = TransitionLayer(num_features, int(num_features * self.compression))

                setattr(self, 'block_transition_' + str(i), transition_layer)

                num_features = int(num_features * self.compression)

            self.dense_trainsition_out_put_list.append(num_features)

        # self.shuortcut_connect_layer = nn.Sequential(nn.Conv3d(self.dense_trainsition_out_put_list[0]*2 + self.dense_trainsition_out_put_list[2], self.dense_trainsition_out_put_list[2],1),
        #                                              InPlaceABNSync(self.dense_trainsition_out_put_list[2]))

        self.shuortcut_connect_layer = nn.Conv3d(self.dense_trainsition_out_put_list[0]*2 + self.dense_trainsition_out_put_list[2], self.dense_trainsition_out_put_list[2],1)

        self.finally_bn = InPlaceABNSync(num_features)
        self.binary_classifier = nn.Linear(num_features, self.number_class)
        self.multiple_classifier = nn.Linear(num_features, 6)


        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)* param.size(4)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm3d) or isinstance(m, InPlaceABNSync) or isinstance(m,InPlaceABN):
        #         m.momentum = 0.01

    def forward(self, x):


        features = self.features(x)

        for i in range(len(self.block_config)):

            features = getattr(self, 'block_' + str(i))(features)

            if i == 0:

                crop_feaure = self.crop_features(features, torch.tensor([8, 8, 8]))

            if i != len(self.block_config) - 1:

                features = getattr(self, 'block_transition_' + str(i))(features)


            if i == 2:

                features = torch.cat((crop_feaure, features),dim=1)
                features = self.shuortcut_connect_layer(features)

        out = self.finally_bn(features)

        out = F.avg_pool3d(out, kernel_size=(8,8,8)).view(features.size(0), -1)

        binary_out   = self.binary_classifier(out)
        multiple_out = self.multiple_classifier(out)

        return binary_out, multiple_out

    def crop_features(self,feature , crop_size):

        feature_size = torch.tensor(feature.shape[2:])

        c = feature_size // 2

        r = crop_size//2

        crop_feature = feature[:,:, c[0]-r[0]:c[0]+r[0], c[1]-r[1]:c[1]+r[1],c[2]-r[2]:c[2]+r[2]]

        return crop_feature


def get_model(type, args):

    if type == "base":

        net = DenseNet()

        loss = Loss(args)

        get_pbb = None

    else:

        raise ("Type Error")

    return args, net, loss, get_pbb



