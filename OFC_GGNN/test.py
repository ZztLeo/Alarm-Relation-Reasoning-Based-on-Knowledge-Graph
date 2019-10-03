import torch
import time
from torch.autograd import Variable

def test(dataloader, model, criterion, optimizer, args):
    test_loss = 0
    correct = 0
    model.eval()
    for i, (alarm, alarm_anno, root_cause_alarm, target) in enumerate(dataloader):
        alarm = Variable(alarm.view(alarm.size()[1]))
        # print(alarm.shape)
        alarm_anno = Variable(alarm_anno)
        # print(alarm_anno)
        root_cause_alarm = Variable(root_cause_alarm)
        # print(alarm)
        # print(root_cause_alarm)
        target = Variable(target)
        # print(target)
        if args.cuda:
            alarm = alarm.cuda()
            alarm_anno = alarm_anno.cuda()
            root_cause_alarm = root_cause_alarm.cuda()
            target = target.cuda()

        output = model(alarm, alarm_anno, root_cause_alarm)
        test_loss += criterion(output, target - 1).item()
        
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred) - 1).cpu().sum()

    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
