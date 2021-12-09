from torch.utils.data import DataLoader

class alarmDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(alarmDataloader, self).__init__(*args, **kwargs)