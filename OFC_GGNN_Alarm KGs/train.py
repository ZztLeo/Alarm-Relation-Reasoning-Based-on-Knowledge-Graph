import torch
import time
from torch.autograd import Variable


def train(epoch, dataloader, model, criterion, optimizer, args):

    model.train()


    for i, (alarm, alarm_anno, root_cause_alarm, target) in enumerate(dataloader):
        model.zero_grad()
        alarm = Variable(alarm.view(alarm.size()[1]))
        #print(alarm.shape)
        alarm_anno = Variable(alarm_anno)
        #print(alarm_anno)
        root_cause_alarm = Variable(root_cause_alarm)
        #print(alarm)
        #print(root_cause_alarm)
        target = Variable(target)
        #print(target)
        if args.cuda:
            alarm = alarm.cuda()
            alarm_anno = alarm_anno.cuda()
            root_cause_alarm = root_cause_alarm.cuda()
            target = target.cuda()

        output = model(alarm, alarm_anno, root_cause_alarm, 'False')
        loss = criterion(output, target-1)
        loss.backward()
        optimizer.step()
        
        if i % int(len(dataloader) / 35 + 1) == 0 and args.print:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch+1, args.niter, i, len(dataloader), loss.item()))
