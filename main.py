from torchvision.transforms import ToTensor, Resize, Normalize, CenterCrop, Compose
from torch.utils.data import DataLoader
from data.dataset import DetailDataset
import matplotlib.pyplot as plt
from models.cnn import CNN
import torch.nn as nn
import torch
import yaml
import os


def train(opt):
    training_data = DetailDataset(
        img_dir=opt['img_dir'],
        train=True,
        transform=Compose([CenterCrop((2800, 2800)),
                           Resize((32, 32)),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                          )
    )

    test_data = DetailDataset(
        img_dir=opt['img_dir'],
        train=False,
        transform=Compose([CenterCrop((2800, 2800)),
                           Resize((32, 32)),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    train_dataloader = DataLoader(training_data, batch_size=opt['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    model = CNN().to(opt['device'])
    print(model)
    loss = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    accuracy, avgloss = [], []
    for i in range(opt['n_epochs']):
        print(f"Epoch: {i}")
        train_loop(train_dataloader, model, loss, optimizer, opt['device'])
        test_loop(test_dataloader, model, loss, opt['device'], accuracy, avgloss)

    plt.plot(accuracy)
    plt.plot(avgloss)
    plt.savefig("checkpoints/accuracyavgloss.png")
    torch.save(model, opt['save_model_name'])


def test(opt):
    test_data = DetailDataset(
        img_dir="data",
        train=False,
        transform=Compose([CenterCrop((2800, 2800)),
                           Resize((32, 32)),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )

    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    model = torch.load(opt['save_model_name'])
    print(model)

    test_loop(test_dataloader, model, nn.BCELoss(), opt['device'], [], [])


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y.type(torch.cuda.FloatTensor))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn, device, accuracy, avgloss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.type(torch.cuda.FloatTensor)).item()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            correct += (pred.type(torch.cuda.LongTensor) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy.append(correct)
    avgloss.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # Read YAML file
    with open("options/opt.yaml", 'r') as stream:
        opt = yaml.safe_load(stream)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if opt["train"]:
        train(opt)
    if opt["test"]:
        test(opt)
