import csv

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

from torchwu.bayes_linear_lrt import BayesLinearLRT
from torchwu.utils.minibatch_weighting import minibatch_weight
from torchwu.utils.variational_approximator import variational_approximator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load / process data
trainset = datasets.MNIST('./data',
                          train=True,
                          download=True,
                          transform=transform)

testset = datasets.MNIST('./data',
                         train=False,
                         download=True,
                         transform=transform)

trainloader = torch.data.utils.DataLoader(trainset,
                                          batch_size=32,
                                          **kwargs)

testloader = torch.data.utils.DataLoader(testset,
                                         batch_size=32,
                                         **kwargs)


@variational_approximator
class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesLinearLRT(input_dim, 512)
        self.blinear2 = BayesLinearLRT(512, output_dim)

    def forward(self, x):
        x_ = x.view(-1, 28 * 28)
        x_ = self.blinear1(x_)
        return self.blinear2(x_)


model = BayesianNetwork(28 * 28, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# prepare results file
with open('results_lrt.csv', 'w+', newline="") as f_out:
    writer = csv.writer(f_out, delimiter=',')
    writer.writerow(['epoch', 'train_loss', 'test_loss', 'accuracy'])

min_test_loss = np.Inf
for epoch in range(20):

    train_loss = 0.0
    test_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(data)

        pi_weight = minibatch_weight(batch_idx=batch_idx, num_batches=32)

        loss = model.elbo(
            inputs=data,
            labels=labels,
            criterion=criterion,
            sample_nbr=5,
            complexity_cost_weight=pi_weight
        )

        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data):05}/{len(trainloader.dataset)} '
                  f'({100 * batch_idx / len(trainloader.dataset):.2f}%)]'
                  f'\tLoss: {loss.item():.6f}')

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)

            loss = model.elbo(
                inputs=data,
                labels=labels,
                criterion=criterion,
                sample_nbr=5
            )

            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += torch.eq(predicted, labels.to).sum().item()

    accuracy = 100 * correct / total
    train_loss /= len(trainloader.dataset)
    test_loss /= len(testloader.dataset)

    if test_loss < min_test_loss:
        print('\nValidation Loss Decreased: {:.6f} -> {:.6f}\n'
              ''.format(min_test_loss, test_loss))

        min_test_loss = test_loss
        torch.save(model.state_dict(), 'mnistBNN_LRT_checkpoint.pt')

    _results = [epoch, train_loss, test_loss, accuracy]

    print(f'Epoch: {epoch:03} | '
          f'Train Loss: {train_loss:.3f} |'
          f'Test Loss: {test_loss:.3f} |'
          f'Accuracy: {accuracy:.3f} %\n')

    # write results to file
    with open('results_lrt.csv', 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
