from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

data_train_transformer = transforms.Compose([transforms.Resize(60),
                                             transforms.RandomCrop(48),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])

data_val_transformer = transforms.Compose([transforms.Resize(60),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])

# 定义dataset 带有图像和标签 | Define dataset class with images and labels
class MyDataset(Dataset):
    def __init__(self, datatxt, datatransform):
        self.images = []
        self.labels = []
        self.datatransform = datatransform
        with open(datatxt, 'r') as f:
            for line in f.readlines():
                item = line.strip().split()
                self.images.append(item[0])
                self.labels.append(int(item[1]))

    def __getitem__(self, index):
        image_path, label = self.images[index], self.labels[index]
        image = Image.open(image_path).convert('RGB')
        image = self.datatransform(image)
        return image, int(label)

    def __len__(self):
        return len(self.images)


'''
data_train_dir = "data/train"  # 定义 train 的路径 | Define the path for the train set
data_val_dir = "data/val"  # 定义 val 的路径 | Define the path for the val set
train_dataset = datasets.ImageFolder(data_train_dir, data_train_transformer)
val_dataset = datasets.ImageFolder(data_val_dir, data_val_transformer)
'''

if __name__ == '__main__':
    train_txt = 'data/train/train.txt'
    val_txt = 'data/test/val.txt'

    train_dataset = MyDataset(train_txt, data_train_transformer)    # MyDataset(all image paths for label a, label a)
    val_dataset = MyDataset(val_txt, data_val_transformer)
    print(train_dataset.__len__())  # Check the number of files for label a
    print(val_dataset.__len__())

    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)   # 下载数据集 | Load dataset
    # DataLoader(dataset, batch_size=images per batch, shuffle=whether to shuffle, num_workers=number of worker threads)
    val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)
