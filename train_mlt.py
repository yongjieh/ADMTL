import torch
import random
import numpy as np
import Task as Task
import trainer as train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(1995)
    task_list = Task.get_Task_List(ids=[0, 1],
                                   set_roots=['data2/1/Art', 'data2/1/real-world'],
                                   batch_size=16)

    trainer = train.Trainer(task_list)
    trainer.train(30)



if __name__ == '__main__':
        main()
