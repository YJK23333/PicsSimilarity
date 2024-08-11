import torch
import Model
import Dataset
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

#prework
Mnist_controller = False
CIFAR_controller = True
device = torch.device("cuda")
loss_history, counter = [],[]
EPOCH = 50
iteration_number = 0
LEARNING_RATE = 0.0005
lowest_model_num = 0
lowest_model_data = 100
if Mnist_controller:
    name = "MNIST"
else:
    if CIFAR_controller:
        name = "CIFAR10"
    else:
        name = "CIFAR100"


#load dataset
if Mnist_controller:
    train_data = torchvision.datasets.MNIST("../dataset",train=True,download=True)
    transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
    sia_train_data = Dataset.Datasets(train_data,transform,False,True,True)
else:
    if CIFAR_controller:
        train_data = torchvision.datasets.CIFAR10("../dataset",train=True,download=True)
        transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        sia_train_data = Dataset.Datasets(train_data,transform,False,True,False)
    else:
        train_data = torchvision.datasets.CIFAR100("../dataset",train=True,download=True)
        transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        sia_train_data = Dataset.Datasets(train_data,transform,False,True,False)



train_dataloader = DataLoader(sia_train_data,batch_size=32,shuffle=True)
net = Model.Sia_net(Mnist_controller).to(device)
loss_fn = Model.ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

#learning step
for epoch in range(0,EPOCH):
    net.train()
    for i, data in enumerate(train_dataloader,0):
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = loss_fn(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Number {} round, Loss : {:.4f}".format(epoch,loss_contrastive.item()))
    if loss_contrastive.item() < lowest_model_data:
        lowest_model_num = epoch
        lowest_model_data = loss_contrastive.item()
        torch.save(net,"Best_{}_one.pth".format(name))

Dataset.show_plot(counter,loss_history)
