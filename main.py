import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import wandb
import omegaconf

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


# Define the training function
def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 99:
            writer.add_scalar('Loss/train', running_loss / 100, epoch * len(train_loader) + batch_idx, epoch)
            writer.add_scalar('Accuracy/train', 100.0 * correct / total, epoch * len(train_loader) + batch_idx, epoch)
            running_loss = 0.0
            correct = 0
            total = 0


@hydra.main(config_path="config/config.yaml", strict=False)
def main(cfg: DictConfig):
    cfg_wandb = omegaconf.OmegaConf.to_container(
        cfg, resolve=True
    )
    wandb.init(project='TestHydraWandb', config=cfg_wandb, sync_tensorboard=True)

    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='/Users/muditverma/work/hydrawandbtest/data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True,
                                               num_workers=2)

    # Create the model
    model = Net(cfg.model.hidden_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    # Set up TensorBoard writer
    writer = SummaryWriter("./tb")
    # Training loop
    for epoch in range(cfg.model.num_epochs):
        print (f"Epoch : {epoch}")
        train(model, train_loader, criterion, optimizer, epoch, writer)

    ## Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
