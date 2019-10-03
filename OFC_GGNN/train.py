import torch
import time
from torch.autograd import Variable


def train(dataloader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    tp = {} # precision
    p = {}  # prediction
    r = {}  # recall

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

        output = model(alarm, alarm_anno, root_cause_alarm)
        loss = criterion(output, target-1)
        loss.backward()
        optimizer.step()
        
        if args.print:
            print('Loss: %.4f' % (loss.item()))
        











class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
