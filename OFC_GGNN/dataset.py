import numpy as np
import torch

class alarmDataset():
    def __init__(self, path, is_train):
        all_data = load_data_from_file(path)
        all_train_data, all_val_data = split_set(all_data)
        print(all_train_data)
        if is_train:
            self.target = all_train_data[:,1]
            self.alarm = all_train_data[:, 0]
            all_train_data = data_convert(all_train_data, 1)
            self.data = all_train_data
            #print(self.data)
        else:
            self.target = all_val_data[:, 1]
            self.alarm = all_val_data[:, 0]
            all_val_data = data_convert(all_val_data, 1)
            self.data = all_val_data
        #print(self.data[5][0])

    def __getitem__(self, index):
        alarm_anno = self.data[index][0]
        #print(alarm_anno)
        alarm_anno = torch.IntTensor(alarm_anno)
        root_cause_alarm = self.data[index][1]
        #print(root_cause_alarm)
        target = self.target[index]
        alarm = self.alarm[index]
        alarm = torch.IntTensor(alarm)
        #print("1")
        #print(alarm.shape)
        return alarm, alarm_anno, root_cause_alarm, target

    def __len__(self):
        return len(self.data)

def load_data_from_file(path):
    label_list = []
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            alarm_list = []
            line_tokens = line.split(" ")
            for i in range(1,len(line_tokens)):
                alarm_list.append(int(line_tokens[i]))
            data_list.append([alarm_list, int(line_tokens[0])])
        f.close()
    return data_list

def split_set(data_list):
    n_examples = len(data_list)
    idx = range(n_examples)
    train = idx[:350]
    val = idx[350:]
    return np.array(data_list)[train], np.array(data_list)[val]

def data_convert(data_list, n_annotation_dim):
    #print(data_list[:,0])
    data_convert_list = []
    max_node_id = find_max_node(data_list[:,0])
    for list in data_list:
        annotation = []
        annotation_target = np.zeros([max_node_id, n_annotation_dim])
        annotation_target[list[1] - 1] = 1
        for i in range(len(list[0])):
            #print(i)
            exec("annotation%s = np.zeros([max_node_id, n_annotation_dim])"%i)
            exec("annotation%s[list[0][i] - 1] = 1"%i)
            #exec("print(annotation%s)"%i)
            #annotation1 = np.zeros([max_node_id, n_annotation_dim])
            #annotation2 = np.zeros([max_node_id, n_annotation_dim])
            exec("annotation.append(annotation%s)"%i)
            #print(annotation)
        #annotation1[list[0][0] - 1] = 1
        #annotation2[list[0][1] - 1] = 1
        #annotation.append([annotation1, annotation2])
        data_convert_list.append([annotation, annotation_target])

    return data_convert_list


def find_max_node(node_list):
    max_node_id = 0
    for node in node_list:
        for i in range(len(node)):
            if node[i] > max_node_id:
                max_node_id = node[i]
    return max_node_id
