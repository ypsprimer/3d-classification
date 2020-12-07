from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


# class Loss(nn.Module):
#
#     def __init__(self, args):
#
#         super(Loss, self).__init__()
#
#         self.args = args
#         self.calculate_multi_label_loss = self.args.calculate_multi_label_loss
#
#     def forward(self, binary_predict, multi_predict, multi_label, train=True):
#
#
#         binary_predict_log_soft_max = F.log_softmax(binary_predict, dim=1)
#         multi_predict_log_soft_max = F.log_softmax(multi_predict, dim=1)
#
#         binary_predict_log_soft_max = binary_predict_log_soft_max.view(binary_predict_log_soft_max.size(0), -1)
#         multi_predict_log_soft_max = multi_predict_log_soft_max.view(multi_predict_log_soft_max.size(0),-1)
#         multi_label   = multi_label.view(-1)
#
#         binary_label = torch.where(multi_label > 0, torch.full_like(multi_label, 1), multi_label).long()
#
#         total_loss = F.nll_loss(binary_predict_log_soft_max, binary_label)
#
#         if self.calculate_multi_label_loss == True:
#
#             total_loss += F.nll_loss(multi_predict_log_soft_max, multi_label)
#
#
#         binary_predict_detach = F.softmax(binary_predict.detach(),dim=1)
#         multi_predict_detach = F.softmax(multi_predict.detach(),dim=1)
#
#         accuracy_ratio = self.calculate_recall_precision(binary_predict_detach, binary_label)
#
#         return total_loss, accuracy_ratio
#
#     def calculate_recall_precision(self, predict_detach, label):
#
#         predict_detach_argmax = torch.argmax(predict_detach, dim=1).long()
#
#         accuracy_number = torch.sum(predict_detach_argmax == label)
#
#         accuracy_ratio = accuracy_number/len(predict_detach)
#
#         return accuracy_ratio


# class Loss(nn.Module):
#
#     def __init__(self, args):
#
#         super(Loss, self).__init__()
#
#         self.args = args
#         self.calculate_multi_label_loss = self.args.calculate_multi_label_loss
#
#     def forward(self, binary_predict, multi_predict, multi_label, train=True):
#
#
#         # binary_predict_log_soft_max = F.log_softmax(binary_predict, dim=1)
#         multi_predict_log_soft_max = F.log_softmax(multi_predict, dim=1)
#
#         binary_predict_soft_max = F.softmax(binary_predict, dim=1)
#         binary_predict_log_soft_max = torch.log(binary_predict_soft_max)
#
#         binary_predict_log_soft_max = binary_predict_log_soft_max.view(binary_predict_log_soft_max.size(0), -1)
#         multi_predict_log_soft_max = multi_predict_log_soft_max.view(multi_predict_log_soft_max.size(0),-1)
#         multi_label   = multi_label.view(-1)
#
#         binary_label = torch.where(multi_label > 0, torch.full_like(multi_label, 1), multi_label).long()
#
#         weight = binary_predict_soft_max
#
#         weight[:, 0] = torch.pow(weight[:,0],2.0)
#         weight[:, 1] = torch.pow(1- weight[:, 1], 2.0)
#
#         binary_predict_log_soft_max  = binary_predict_log_soft_max * weight
#
#         total_loss = F.nll_loss(binary_predict_log_soft_max, binary_label)
#
#         if self.calculate_multi_label_loss == True:
#
#             total_loss += F.nll_loss(multi_predict_log_soft_max, multi_label)
#
#
#         binary_predict_detach = F.softmax(binary_predict.detach(),dim=1)
#         multi_predict_detach = F.softmax(multi_predict.detach(),dim=1)
#
#         binary_accuracy_ratio = self.calculate_recall_precision(binary_predict_detach, binary_label)
#
#         # print("**********************************************************************")
#         # print("label ", binary_label)
#         # print("pre " , binary_predict_detach)
#         # print("precision", binary_accuracy_ratio)
#         # print("loss ", total_loss)
#
#         return total_loss, binary_accuracy_ratio
#
#     def calculate_recall_precision(self, predict_detach, label):
#
#         predict_detach_argmax = torch.argmax(predict_detach, dim=1).long()
#
#         accuracy_number = torch.sum(predict_detach_argmax == label)
#
#         positive_accuracy =  torch.sum((predict_detach_argmax == 1)*(label==1))
#         negative_accuracy =  torch.sum((predict_detach_argmax == 0)*(label==0))
#
#         positive_accuracy_number = torch.sum(label)
#         negative_accuracy_number = len(label) - positive_accuracy_number
#
#         if positive_accuracy_number != 0:
#
#             positive_accuracy = positive_accuracy.float() / positive_accuracy_number
#
#         else:
#
#             positive_accuracy = torch.from_numpy(np.array([np.nan],dtype=np.float32)).cuda()[0]
#
#
#         if negative_accuracy_number != 0:
#
#             negative_accuracy = negative_accuracy.float() / negative_accuracy_number
#
#         else:
#
#             negative_accuracy = torch.from_numpy(np.array([np.nan],dtype=np.float32)).cuda()[0]
#
#
#         accuracy_ratio = accuracy_number.float()/len(predict_detach)
#
#         return accuracy_ratio, positive_accuracy, negative_accuracy

class Loss(nn.Module):

    def __init__(self, args):

        super(Loss, self).__init__()

        self.args = args
        self.calculate_multi_label_loss = self.args.calculate_multi_label_loss

    def forward(self, binary_predict, multi_predict, multi_label, train=True, binary_out2=None):

        #print(multi_label.cpu().numpy())
        binary_predict_soft_max = F.softmax(binary_predict, dim=1)
        binary_predict_log_soft_max = torch.log(binary_predict_soft_max)
        binary_predict_log_soft_max = binary_predict_log_soft_max.view(binary_predict_log_soft_max.size(0), -1)
        binary_label = torch.sum(multi_label, dim=1).long()


        weight = torch.pow(1 - binary_predict_soft_max, 2.0)
        binary_loss = F.nll_loss(binary_predict_log_soft_max, binary_label)
        binary_predict_detach = binary_predict_soft_max.detach()
        binary_accuracy_ratio = self.calculate_recall_precision(binary_predict_detach, binary_label)
        wrong_index = self.calculate_wrong_crop_image(binary_predict_detach, binary_label)

        '''
        multi_predict_p = F.sigmoid(multi_predict)
        multi_predict_detach = multi_predict_p.detach()
        multi_predict_p = torch.transpose(multi_predict_p, 0, 1)
        multi_predict_p_r = (torch.max(multi_predict_p, dim=0, keepdim=True)[0] - 0.00001) * torch.ones_like(multi_predict_p)
        multi_predict_p = torch.stack([multi_predict_p_r, multi_predict_p], dim=-1)
        multi_label_T = torch.transpose(multi_label, 0, 1)
        multi_loss = - 5 * F.logsigmoid(multi_predict) * multi_label.float() + (- F.logsigmoid(-multi_predict) * (1 - multi_label.float()))
        total_loss = torch.sum(multi_loss) / multi_loss.shape[0]  #+ binary_loss
        '''


        multi_predict_p = F.softmax(multi_predict, dim=1)
        multi_predict_detach = multi_predict_p.detach()
        multi_predict_p = torch.transpose(multi_predict_p, 0, 1)
        multi_predict_p = torch.stack([1 - multi_predict_p, multi_predict_p], dim=-1)
        multi_label_T = torch.transpose(multi_label, 0, 1)

        multiple_loss = F.nll_loss(F.log_softmax(multi_predict, dim=1), torch.argmax(multi_label, dim=1), reduction='none')
        multiple_loss = torch.sum(multiple_loss * binary_label.float()) / (torch.sum(binary_label.float()) + 1e-4)

        multi_accuracy_ratio = []
        for multi_predict_p_x, multi_label_T_x in zip(multi_predict_p, multi_label_T):
            multi_accuracy_ratio.append(self.calculate_recall_precision(multi_predict_p_x, multi_label_T_x))

        total_loss = multiple_loss

        return total_loss, [binary_accuracy_ratio] + multi_accuracy_ratio, wrong_index, multi_predict_detach

    def calculate_wrong_crop_image(self, binary_predict_detach, binary_label):

        predict_detach_argmax = torch.argmax(binary_predict_detach, dim=1).long()

        wrong_index = (predict_detach_argmax != binary_label)

        return wrong_index

    def calculate_recall_precision(self, predict_detach, label):

        predict_detach_argmax = torch.argmax(predict_detach, dim=1).long()

        accuracy_number = torch.sum(predict_detach_argmax == label)

        positive_accuracy = torch.sum((predict_detach_argmax == 1) * (label == 1))
        negative_accuracy = torch.sum((predict_detach_argmax == 0) * (label == 0))

        positive_accuracy_number = torch.sum(label)
        negative_accuracy_number = len(label) - positive_accuracy_number

        if positive_accuracy_number != 0:

            positive_accuracy = positive_accuracy.float() / positive_accuracy_number

        else:

            positive_accuracy = torch.from_numpy(np.array([np.nan], dtype=np.float32)).cuda()[0]

        if negative_accuracy_number != 0:

            negative_accuracy = negative_accuracy.float() / negative_accuracy_number

        else:

            negative_accuracy = torch.from_numpy(np.array([np.nan], dtype=np.float32)).cuda()[0]

        accuracy_ratio = accuracy_number.float() / len(predict_detach)

        return accuracy_ratio, positive_accuracy, negative_accuracy

class ClassificationNoiseCorrectionLoss(nn.Module):

    def __init__(self):

        super(ClassificationNoiseCorrectionLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predict, binary_label):

        loss = self.ce_loss(predict, binary_label)

class NoiseCorrectionLoss(nn.Module):

    def __init__(self, args, label_number=10):

        super(NoiseCorrectionLoss, self).__init__()

        self.big_table = nn.parameter(torch.zeros((label_number,2)))
        self.args = args
        self.classification_loss = ClassificationNoiseCorrectionLoss()


    def forward(self, binary_predict, multiple_truth_label, index, epoch):

        if epoch < self.args.stage1:

            #lc is classification loss
            lc = self.classification_loss()

        else:

            pass

        current_virtual_label_probability = torch.index_select(self.big_table, 0, index)

        current_virtual_label_probability = F.softmax(current_virtual_label_probability)

        classifacation_loss = None
        compatibility_loss  = None
        entropy_loss        = None


        return classifacation_loss + compatibility_loss + entropy_loss
