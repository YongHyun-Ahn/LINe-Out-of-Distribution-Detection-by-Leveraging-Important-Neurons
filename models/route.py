import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class RouteLUNCH(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p_w=10, p_a=10, conv1x1=False, info=None, clip_threshold = 1e10):
        super(RouteLUNCH, self).__init__(in_features, out_features, bias)
        self.p = p_a
        self.weight_p = p_w
        self.clip_threshold = clip_threshold
        self.info = info
        self.masked_w = None
        self.mask_f = None
        self.l_weight = self.weight.data.cuda()

    def calculate_shap_value(self):
        self.contrib = self.info.T
        self.mask_f = torch.zeros(self.out_features,self.in_features)
        self.masked_w = torch.zeros((self.out_features,self.out_features,self.in_features))

        for class_num in range(self.out_features):
            self.matrix = abs(self.contrib[class_num,:]) * self.weight.data.cpu().numpy()
            self.thresh = np.percentile(self.matrix, self.weight_p)
            mask_w = torch.Tensor((self.matrix > self.thresh))
            self.masked_w[class_num,:,:] = (self.weight.squeeze().cpu() * mask_w).cuda()
            self.class_thresh = np.percentile(self.contrib[class_num,:], self.p)
            self.mask_f[class_num,:] = torch.Tensor((self.contrib[class_num,:] > self.class_thresh))

    def forward(self, input):    
        if self.masked_w is None:
            self.calculate_shap_value()
        pre = input[:, None, :] * self.weight.data.cuda()
        if self.bias is not None:
            pred = pre.sum(2) + self.bias
        else:
            pred = pre.sum(2)
        pred = torch.nn.functional.softmax(pred, dim=1)   
        preds = np.argmax(pred.cpu().detach().numpy(), axis=1)
       
        counter_cp = 0
        cp = torch.zeros((len(input), self.in_features)).cuda()
        for idx in preds:
            cp[counter_cp,:] = input[counter_cp,:] * self.mask_f[idx,:].cuda()     
            counter_cp = counter_cp + 1

        vote = torch.zeros((len(preds),self.out_features,self.in_features)).cuda()
        counter_dice = 0
        for idx in preds:
            vote[counter_dice,:,:] = cp[counter_dice,:] * self.masked_w[idx,:,:].cuda()
            counter_dice = counter_dice + 1
        
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)    
        return out