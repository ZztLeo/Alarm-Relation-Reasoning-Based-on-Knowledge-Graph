import numpy as np
import random

def create_trainset(file_name):
    data = np.zeros((500,5))
    with open(file_name, 'r') as f:
        lines = f.readlines()
        a = np.zeros((23, 5))
        line_count = 0
        for line in lines:
            line = line.strip().split(' ')
            a[line_count, : len(line)] = line[:]
            line_count += 1
        for i in range(500):
            j = random.randint(0,22)
            k = random.randint(1,4)
            l = random.randint(1,4)
            m = a[j,k]
            n = a[j,l]
            a[j, k] = n
            a[j, l] = m
            data[i,:] = a[j,:]
        np.random.shuffle(data)
        print(data)
    np.savetxt('create_trainset.txt',data,fmt='%d')



def main():
    file_name = "E:/graduate_student/Lab/GGNN故障图谱仿真/OFC_GGNN/alarm_data/train_test.txt"
    create_trainset(file_name)

if __name__ == "__main__":
    main()