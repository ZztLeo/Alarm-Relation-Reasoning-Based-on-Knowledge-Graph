import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
from GGNN import GGNN
from args import args


class GRM(nn.Module):
    def __init__(self, adjacency_matrix, num_class = 4, ggnn_hidden_channel = 40, ggnn_output_channel = 20,
                 time_step = 20, alarm_num = 14):
        super(GRM, self).__init__()
        self._num_class = num_class
        self._ggnn_hidden_channel = ggnn_hidden_channel
        self._ggnn_output_channel = ggnn_output_channel
        self._time_step = time_step
        self._adjacency_matrix = adjacency_matrix
        self._alarm_num = alarm_num
        self._totalnode_num = num_class + alarm_num

        self.ggnn = GGNN(hidden_state_channel = self._ggnn_hidden_channel,
			output_channel = self._ggnn_output_channel,
			time_step = self._time_step,
			adjacency_matrix=self._adjacency_matrix,
			num_classes = self._num_class)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self._ggnn_output_channel * (self._alarm_num + 1), 14),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(14, 1)
        )

        self.ReLU = nn.ReLU(True)

    def forward(self, alarm, alarm_anno, root_cause_alarm, is_train, ifprint = False):
        #print(alarm, root_cause_alarm)
        contextual = Variable(torch.zeros(self._totalnode_num, self._ggnn_hidden_channel))
        #print(ifprint)
        '''if args.cuda:
            contextual.cuda()'''
        contextual[0:self._num_class, 0] = 1.  # 加前缀
        contextual[self._num_class:, 1] = 1 # 加前缀
        #print("1")
        #print("alarm:\n",alarm)
        annotation_entity1 = Variable(torch.zeros([1, 18]))
        annotation_entity1[0,0] = 1
        annotation_entity2 = Variable(torch.zeros([1, 18]))
        annotation_entity2[0,1] = 1
        annotation_entity3 = Variable(torch.zeros([1, 18]))
        annotation_entity3[0,2] = 1
        annotation_entity4 = Variable(torch.zeros([1, 18]))
        annotation_entity4[0,3] = 1
        contextual[0:1, 2:20] = annotation_entity1[:]
        contextual[1:2, 2:20] = annotation_entity2[:]
        contextual[2:3, 2:20] = annotation_entity3[:]
        contextual[3:4, 2:20] = annotation_entity4[:]
        #print("contextual:\n",contextual)
        alarm_anno = alarm_anno.view(-1, self._totalnode_num)
        root_cause_alarm = root_cause_alarm.view(-1, self._totalnode_num)
        #print(root_cause_alarm)
        #print("root_cause_alarm:\n",root_cause_alarm)
        #print(contextual)
        #print(alarm_anno)
        for i in range(len(alarm)-1):
            #print(len(alarm)-1)
           
            contextual[int(alarm[i]-1), 2:20] = alarm_anno[i,:]

        #print(contextual)
        #if is_train:
        #contextual[0:self._num_class, 2:] = root_cause_alarm[:]
        #print("contextual:\n",contextual)
        #print("contextual.shape:",contextual.shape)
        ggnn_input = contextual

        ggnn_feature = self.ggnn(ggnn_input, ifprint)

        ggnn_feature_norm = ggnn_feature.view(self._num_class, -1)
        #print(ggnn_feature_norm.shape)
        # classifier
        final_scores = self.classifier(ggnn_feature_norm).view(1,-1)
        #print(final_scores)
        return final_scores


    def _initialize_weights(self):
        for m in self.classifier.modules():
            cnt = 0
            if isinstance(m, nn.Linear):
                if cnt == 0:
                    m.weight.data.normal_(0, 0.001)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                cnt += 1
