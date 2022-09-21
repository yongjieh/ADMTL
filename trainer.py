from misc import AverageMeter
import torch.nn.functional as F
import torch
import numpy as np
import copy


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Trainer(object):
    def __init__(self, task_list):

        self.task_list = task_list

        self.scheduler = None

        self.empty_tasks = set()

        self.steps = 0
        self.e_steps = 0
        self.best_acc = [0 for _ in range(len(task_list))]
        self.best_epoch = [0 for _ in range(len(task_list))]
        self.loss = [[] for _ in range(len(task_list))]

    def train(self, n_epoch):

        for i_epoch in range(n_epoch):

            lr_decay_epoch = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
            lr_decay_count = 0
            lr_decay = 0.5
            if i_epoch == lr_decay_epoch[lr_decay_count]:
                for task in self.task_list:
                    for param_group in task.optim.param_groups:
                        param_group['lr'] = param_group['lr'] * lr_decay
                    lr_decay_count += 1

            self._train_epoch()  # 训练

            dev_acc, dev_loss = self._eval_epoch()  # 测试

            for k, acc in enumerate(dev_acc):
                if acc.avg > self.best_acc[k]:
                    self.best_acc[k] = acc.avg
                    self.best_epoch[k] = i_epoch
                self.loss[k].append(dev_loss[k].avg)

        # np.savetxt('loss.txt', self.loss)
        for task in self.task_list:
            #save_path = 'test_' + str(task.task_id) + '.pth'
            #torch.save(task.net.state_dict(), save_path)
            print('task_id:%d bestacc:%.5f epoch: %d' % (
                task.task_id, self.best_acc[task.task_id], self.best_epoch[task.task_id]))

    def _train_epoch(self):
        n_tasks = len(self.task_list)

        task_seq = list(np.random.permutation(n_tasks))
        empty_task = copy.deepcopy(self.empty_tasks)

        totoal_loss = [AverageMeter() for _ in range(n_tasks)]
        totoal_acc = [AverageMeter() for _ in range(n_tasks)]

        for task in self.task_list:
            task.net.train()

        for task in self.task_list:
            task.train_data_loader = iter(task.train_data_loader)

        print('epoch:%d' % (self.steps + 1))
        while len(empty_task) < n_tasks:
            for task_id in task_seq:
                if task_id in empty_task:
                    continue
                task = self.task_list[task_id]
                batch = next(task.train_data_loader, None)
                if batch is None:
                    empty_task.add(task_id)
                    task.init_train_loader()
                    continue

                task.net.zero_grad()

                for p in task.net.parameters():
                    p.requires_grad = True

                x, y = batch
                batch_x = x.cuda()
                batch_y = y.cuda()

                out, mmdlosses, masklosses = task.net(batch_x, task.mlts, task.pre)

                loss = F.cross_entropy(out, batch_y) + mmdlosses + masklosses

                pred, = accuracy(out, batch_y, (1,))

                totoal_loss[task_id].update(loss.item(), x.size(0))
                totoal_acc[task_id].update(pred, x.size(0))

                torch.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=True)
                task.optim.step()
                task.optim.zero_grad()

                for p in task.net.parameters():
                    p.requires_grad = False

        self.steps += 1

        for task in self.task_list:
            print('acc_%d:%.5f' % (task.task_id, totoal_acc[task.task_id].avg))
            print('loss_%d:%.5f' % (task.task_id, totoal_loss[task.task_id].avg))

    def _eval_epoch(self):
        n_tasks = len(self.task_list)
        totoal_loss = [AverageMeter() for _ in range(n_tasks)]
        totoal_acc = [AverageMeter() for _ in range(n_tasks)]

        for task in self.task_list:
            task.net.eval()

        with torch.no_grad():
            for k in range(n_tasks):
                task = self.task_list[k]
                test_dataloader = task.test_data_loader
                for i, batch in enumerate(test_dataloader):
                    x, y = batch
                    batch_x = x.cuda()
                    batch_y = y.cuda()
                    out, _, _ = task.net(batch_x, task.mlts,task.pre)

                    loss = F.cross_entropy(out, batch_y)

                    pred, = accuracy(out, batch_y, (1,))
                    totoal_loss[task.task_id].update(loss.item(), x.size(0))
                    totoal_acc[task.task_id].update(pred, x.size(0))

            self.e_steps += 1

            print('[%d]' % (self.e_steps))
            for task in self.task_list:
                print('acc_%d:%.5f' % (task.task_id, totoal_acc[task.task_id].avg))
                print('loss_%d:%.5f' % (task.task_id, totoal_loss[task.task_id].avg))

        return totoal_acc, totoal_loss
