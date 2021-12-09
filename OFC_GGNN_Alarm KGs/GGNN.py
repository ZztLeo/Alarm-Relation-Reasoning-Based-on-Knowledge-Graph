import os,sys
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from args import args



class GGNN(nn.Module):
    def __init__(self,adjacency_matrix, hidden_state_channel=20, output_channel=8,
				time_step = 5, num_classes = 4):
        super(GGNN, self).__init__()
        self.time_step = time_step
        self.hidden_state_channel = hidden_state_channel
        self.output_channel = output_channel
        self.adjacency_matrix = adjacency_matrix
        self.num_classes = num_classes
        self.alarm_num = 14

        self._in_matrix, self._out_matrix = self.split_matrix(self.adjacency_matrix)
        self._in_matrix = Variable(torch.from_numpy(self._in_matrix), requires_grad=False)
        self._out_matrix = Variable(torch.from_numpy(self._out_matrix), requires_grad=False)
        #print(self._in_matrix)
        self._mask = Variable(torch.zeros(self.num_classes, self.alarm_num),requires_grad=False)
        #print(self._mask)
        self._mask = self._in_matrix[0:self.num_classes, self.num_classes:]
        #print(self._in_matrix[0:self.num_classes, self.alarm_num:])
        #ggnn layer(等式序号与原GGNN相符合)
        self.fc_eq3_wz = nn.Linear(2 * hidden_state_channel, hidden_state_channel)
        self.fc_eq3_uz = nn.Linear(hidden_state_channel, hidden_state_channel)
        self.fc_eq4_wr = nn.Linear(2 * hidden_state_channel, hidden_state_channel)
        self.fc_eq4_ur = nn.Linear(hidden_state_channel, hidden_state_channel)
        self.fc_eq5_w = nn.Linear(2 * hidden_state_channel, hidden_state_channel)
        self.fc_eq5_u = nn.Linear(hidden_state_channel, hidden_state_channel)

        self.fc_output = nn.Linear(2 * hidden_state_channel, output_channel)
        self.ReLU = nn.ReLU(True)

        #self.test_fc = nn.Linear(output_channel, 1)
        self.reason_fc_x = nn.Linear(hidden_state_channel, output_channel)
        self.reason_fc_y = nn.Linear(hidden_state_channel, output_channel)
        self.reason_fc2 = nn.Linear(output_channel, 1)

        self._initialize_weights()

        '''if args.cuda:
            self._in_matrix.cuda()
            self._out_matrix.cuda()
            self._mask.cuda()'''


    def forward(self, input, ifprint = False):
        input = input.view(-1, self.hidden_state_channel) #view相当于resize，-1表示根据另一参数进行推测

        node_num = self._in_matrix.size()[0]

        h_operation = input.view(node_num, self.hidden_state_channel)
        #h_operation = torch.tensor(h_operation, dtype=torch.float64).cuda()
        #print("h_operation:\n", h_operation)
        #print("h_operation.shape:", h_operation.shape)
        h_in_matrix = self._in_matrix.view(node_num, -1)
        #h_in_matrix = torch.tensor(h_in_matrix, dtype=torch.float32)
        h_out_matrix = self._out_matrix.view(node_num, -1)
        #print("h_in_matrix:\n", h_in_matrix)
        #print("h_in_matrix.shape:", h_in_matrix.shape)
        #print("h_out_matrix:\n", h_out_matrix)
        #print("h_out_matrix.shape:", h_out_matrix.shape)
        #propogation process
        for step in range(self.time_step):
            h_operation = torch.tensor(h_operation, dtype=torch.float64)
            '''if args.cuda:
                self.h_operation.cuda()'''

            #print("h_operation:\n", h_operation)
            #print("h_operation.shape:", h_operation.shape)

            #print("h_in_matrix:\n", h_in_matrix)
            #print("h_in_matrix.shape:", h_in_matrix.shape)
            #print("h_out_matrix:\n", h_out_matrix)
            #print("h_out_matrix.shape:", h_out_matrix.shape)

            #eq(2)
            a_v = torch.cat((torch.mm(h_in_matrix, h_operation), torch.mm(h_out_matrix, h_operation)), 1)
            #print("a_v:\n",a_v)
            hidden_state_v = h_operation.view(node_num, -1)
            #print("hidden_state_v:\n",hidden_state_v)
            a_v = a_v.float()
            hidden_state_v = hidden_state_v.float()
            # eq(3)  update gate
            z_v = torch.sigmoid(self.fc_eq3_wz(a_v) + self.fc_eq3_uz(hidden_state_v))

  
            # eq(4)  reset gate
            r_v = torch.sigmoid(self.fc_eq4_wr(a_v) + self.fc_eq4_ur(hidden_state_v))

            # eq(5)
            h_v_generated = torch.tanh(self.fc_eq5_w(a_v) + self.fc_eq5_u(r_v * hidden_state_v))

            # eq(6)
            hidden_state_v = (1 - z_v) * hidden_state_v + z_v * h_v_generated

            h_operation = hidden_state_v.view(node_num, -1)

        # 每个节点/图输出：	ov = g(h(T)v，xv)，其中g是函数，表示利用逐个nodes的最终状态和初始输入分别求输出
        #print("hidden_state_v:\n",hidden_state_v)
        output = torch.cat((hidden_state_v, input), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        #print("output:\n",output)
        #print(ifprint)
        
        dist1 = np.sqrt(np.sum(np.square(output[15,:].detach().numpy() - output[16,:].detach().numpy() )))
        dist2 = np.sqrt(np.sum(np.square(output[5,:].detach().numpy() - output[11,:].detach().numpy() )))
        dist3 = np.sqrt(np.sum(np.square(output[8,:].detach().numpy() - output[9,:].detach().numpy() )))
        
        #print("dist1:\n", dist1)
        #print("dist2:\n", dist2)
        #print("dist3:\n", dist3)
            #print(output[15,:])
            #print(output[16,:])
        #print("output.shape:",output.shape)
        
        ####注意力机制###
        h_atten = hidden_state_v.view(node_num, -1)
        alarm_num = self.alarm_num
        cnode = h_atten[0:self.num_classes, :].contiguous().view(-1, self.hidden_state_channel)  # root cause node
        cfcx = torch.tanh(self.reason_fc_x(cnode))
        cnode_enlarge = cfcx.contiguous().view(self.num_classes, 1, -1).repeat(1, alarm_num, 1)

        anode = h_atten[self.num_classes:, :].contiguous().view(-1, self.hidden_state_channel)  # alarm node
        afcy = torch.tanh(self.reason_fc_y(anode))
        anode_enlarge = afcy.contiguous().view(1, alarm_num, -1).repeat(1, self.num_classes, 1, 1)

        cacat = (cnode_enlarge.contiguous().view(-1, self.output_channel)) * (
            anode_enlarge.contiguous().view(-1, self.output_channel))
   
        rfc = self.reason_fc2(cacat)
        rfc = torch.sigmoid(rfc)
        
        #print("rfc:",rfc)
        #print("maskshape:",self._mask)
        rfc = rfc.float()    
        mask_enlarge = self._mask.repeat(1, 1).view(-1, 1)
        #print("maskshape:",mask_enlarge)
        mask_enlarge = mask_enlarge.float()
        rfc = rfc * mask_enlarge

        output = output.contiguous().view(node_num, -1)
        routput = output[0: self.num_classes, :]
        ooutput = output[self.num_classes:, :]
        ooutput_enlarge = ooutput.contiguous().view(1, -1).repeat(self.num_classes, 1).view(-1,
                                                                                                           self.output_channel)
        weight_ooutput = ooutput_enlarge * rfc
        weight_ooutput = weight_ooutput.view(self.num_classes,self.alarm_num, -1)

        final_output = torch.cat((routput.contiguous().view(self.num_classes, 1, -1), weight_ooutput), 1)
        #print("final_output:\n",final_output)
        #print("final_output.shape:",final_output.shape)
        return final_output

    def _initialize_weights(self):
        for m in self.reason_fc2.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()
        for m in self.reason_fc_x.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        for m in self.reason_fc_y.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def split_matrix(self,matrix_file):
        d_row, d_col = matrix_file.shape

        in_matrix = np.zeros((d_row, d_col))
        in_matrix = matrix_file
        out_matrix = np.zeros((d_col, d_row))
        out_matrix = matrix_file.transpose()

        return in_matrix, out_matrix

