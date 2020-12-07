import torch
from torch import nn
from torch.autograd import Variable

class Wraper(nn.Module):

    def __init__(self, net, loss_fun=None):

        super(Wraper, self).__init__()

        self.net = net
        self.loss_fun = loss_fun


    def forward(self, data, label, train_stage = True):

        binary_out, multi_out, binary_out2 = self.net(data)

        if train_stage == True:
            total_loss_list = []
            accuracy_ratio_list = []
            binary_predict_detach_list = []
            total_loss, accuracy_ratio, wrong_index, binary_predict_detach = self.loss_fun(binary_out, multi_out, label)

            if binary_out2 is not None:
                for idx, binary_out2_x in enumerate(binary_out2):
                    total_loss2, accuracy_ratio2, _, binary_predict_detach2 = self.loss_fun(binary_out2_x, multi_out, label)
                    total_loss_list.append(total_loss2)
                    accuracy_ratio_list.append(accuracy_ratio2)
                    binary_predict_detach_list.append(binary_predict_detach2)
                accuracy_ratio_unit = self.loss_fun.calculate_recall_precision(torch.stack(binary_predict_detach_list).mean(dim=0), label.view(-1).long())
                accuracy_ratio_list.append(accuracy_ratio_unit)
            return [total_loss] + total_loss_list, [accuracy_ratio] + accuracy_ratio_list, \
                   wrong_index, [binary_predict_detach] + binary_predict_detach_list

        return binary_out, multi_out, binary_out2