import torchvision.models.resnet
from PIL import Image
import os
import os.path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset


class DataSet(VisionDataset):

    def __init__(self, root, transform_common=None, transform=None, target_transform=None, transforms=None, train=True):
        super(DataSet, self).__init__(root, transforms, transform, target_transform)
        path = root
        imgs = []
        files = os.listdir(path)
        files.sort()
        idx = 0
        for file in files:
            classfiles = os.path.join(root, file)
            classimages = os.listdir(classfiles)
            length = len((classimages))
            if train:
                classimages = classimages[:int(0.8 * length)]
            else:
                classimages = classimages[int(0.8 * length):]
            for classimage in classimages:
                imgs.append((os.path.join(classfiles, classimage), idx))
            idx = idx + 1
        self.imgs = imgs
        self.transform_common = transform_common

    def __getitem__(self, index):
        """
        Args:
            index (int): Index.convert('L')

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        #self.imgs[index][0]
        img = Image.open(self.imgs[index][0]).convert('RGB')
        if self.transform_common is not None:
            img = self.transform_common(img)
        return img, self.imgs[index][1]

    def __len__(self):
        return len(self.imgs)


def get_train_loader(root, batch_size):
    dataset = DataSet(root, transform_common=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ]))
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader


def get_test_loader(root, batch_size):
    dataset = DataSet(root, transform_common=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ]), train=False)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader


if __name__ == '__main__':
    train_dataloader = get_train_loader('../data/data2/1/a', 128)
    tt = iter(train_dataloader)
    while True:
        out = next(tt, None)
        print(out)

    for i, data in enumerate(train_dataloader):
        print(i)
        pass;
