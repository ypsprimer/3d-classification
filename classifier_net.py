import torch
# from modules import InPlaceABN, InPlaceABNSync
from modules import InPlaceABN as InPlaceABNSync

from torch import nn
import torch.nn.functional as F
from losser import Loss


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = InPlaceABNSync(n_in)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = InPlaceABNSync(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride))

        else:
            self.shortcut = None

    def forward(self, x):
        residual = x.clone()
        if self.shortcut is not None:
            #             print('here')
            residual = self.shortcut(residual)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)

        out += residual
        return out

class Net(nn.Module):

    def __init__(self, args):

        super(Net, self).__init__()

        # self.preBlock = nn.Sequential(
        #     nn.Conv3d(1, 32, kernel_size = (1 ,3 ,3), padding = (0 ,1 ,1)),
        #     InPlaceABNSync(32),
        #     nn.Conv3d(32, 32, kernel_size = (1 ,3 ,3), padding = (0 ,1 ,1)),
        #     InPlaceABNSync(32),
        #     # nn.MaxPool3d(kernel_size=(2,2,2),stride=(2, 1, 1),return_indices=False), #add by lxw
        #     nn.Conv3d(32, 32, kernel_size = (1 ,3 ,3), padding = (0 ,1 ,1)),
        #     InPlaceABNSync(32))
        self.args = args
        self.multiple_channel = self.args.multiple_channel

        self.preBlock = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size = (3 ,3 ,3), padding = (1 ,1 ,1)),
            InPlaceABNSync(32),
            nn.Conv3d(32, 32, kernel_size = (3 ,3 ,3), padding = (1 ,1 ,1)),
            InPlaceABNSync(32),
            nn.Conv3d(32, 32, kernel_size = (3 ,3 ,3), padding = (1 ,1 ,1)),
            InPlaceABNSync(32),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            InPlaceABNSync(32))

        num_blocks_forw = [4 ,4 ,8 ,8]
        self.featureNum_forw = [32 ,32 ,64 ,64 ,64]


        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[ i +1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[ i +1], self.featureNum_forw[ i +1]))
            print('forw' + str(i + 1), self.featureNum_forw[i], self.featureNum_forw[ i +1])
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.maxpool_2d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)

        self.drop = nn.Dropout3d(p=0.5, inplace=False)

        self.average_pooling = torch.nn.AvgPool3d((3,3,3),stride=1)

        self.dense_connect_cnn = torch.nn.Sequential(
            torch.nn.Linear(64 * 1 * 1 * 1, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU()
        )

        self.binary_classification_out = torch.nn.Linear(128, 2)
        self.multi_classification_out = torch.nn.Linear(128, self.multiple_channel)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, InPlaceABNSync):
                m.momentum = 0.01

    def forward(self, x):
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool_2d(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool(out1)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool(out3)
        out4 = self.forw4(out3_pool)  # 96

        average_pooling_out = self.average_pooling(out4)

        average_pooling_out = average_pooling_out.view(average_pooling_out.size(0), -1)

        cnn_connect = self.dense_connect_cnn(average_pooling_out)

        binary_out = self.binary_classification_out(cnn_connect)
        multi_out = self.multi_classification_out(cnn_connect)

        return binary_out, multi_out


def get_model(type, args):

    if type == "base":

        net = Net(args)

        loss = Loss(args)

        get_pbb = None

    else:

        raise ("Type Error")

    return args, net, loss, get_pbb