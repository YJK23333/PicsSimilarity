import torch
import Dataset
import torchvision.datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

Mnist_controller = False
CIFAR_controller = True
if Mnist_controller:
    name = "MNIST"
else:
    if CIFAR_controller:
        name = "CIFAR10"
    else:
        name = "CIFAR100"

if Mnist_controller:
    test_data = torchvision.datasets.MNIST("../dataset",train=False,download=True)
    transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
    sia_test_data = Dataset.Datasets(test_data,transform,False,False,True)
else:
    if CIFAR_controller:
        test_data = torchvision.datasets.CIFAR10("../dataset",train=False,download=True)
        transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        sia_test_data = Dataset.Datasets(test_data,transform,False,False,False)
    else:
        test_data = torchvision.datasets.CIFAR100("../dataset",train=False,download=True)
        transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        sia_test_data = Dataset.Datasets(test_data,transform,False,False,False)



test_dataloader = DataLoader(sia_test_data,batch_size=1,shuffle=True)

net = torch.load("Best_{}_one.pth".format(name))

for data in test_dataloader:
    x0, x1, label = data
    concatenated = torch.cat((x0,x1),0)
    output1,output2 = net(x0.cuda(),x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    Dataset.imshow(torchvision.utils.make_grid(concatenated),
                   "Dissimilarity: {:.2f}".format(euclidean_distance.item()))

