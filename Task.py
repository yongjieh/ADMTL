import dataset.dataloader as dataloader
import model.mltvgg as vgg
import torchvision.models as pmodels
import torch


def get_Task_List(ids, set_roots, batch_size):
    task_list = []
    models = {}
    num_class = 10
    pre_model = pmodels.vgg11(pretrained=True)
    for p in pre_model.parameters():
        p.requires_grad = False
    pre_model.cuda()
    for i, id in enumerate(ids):
        model = vgg.mlt_vgg(num_class)

        for p in model.parameters():
            p.requires_grad = False
        models[id] = model

        task = Task(id, set_roots[i], model, pre_model.features)
        task.init_data_loader(batch_size)#初始化数据集
        task_list.append(task)
    for task in task_list:
        #初始化网络和多任务网络
        t_mlts = []
        for m_id in models.keys():
            if not m_id == task.task_id:
                t_mlts.append(models[m_id])
        task.mlts = t_mlts
        task.net.init(pre_model.features, t_mlts)
        #task.net.load_state_dict(torch.load('test_0.pth'))
        task.net.cuda()
        optimizer = torch.optim.Adam(task.net.parameters(), lr=0.001)
        task.optim = optimizer
    return task_list


class Task(object):
    def __init__(self, task_id, set_root, model, pre, mlts=None):
        self.task_id = task_id

        self.pre = pre

        self.mlts = mlts

        self.net = model

        self.optim = None

        self.set_root = set_root

        self.train_data_loader = None
        self.test_data_loader = None

        self.batch_size = None

    def init_train_loader(self):
        self.train_data_loader = dataloader.get_train_loader(self.set_root, self.batch_size)

    def init_data_loader(self, batch_size):
        self.batch_size = batch_size
        self.train_data_loader = dataloader.get_train_loader(self.set_root, batch_size)
        self.test_data_loader = dataloader.get_test_loader(self.set_root, batch_size)
