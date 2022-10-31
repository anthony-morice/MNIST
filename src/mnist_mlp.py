import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys

BATCH_SIZE = 64
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.model = nn.Sequential(
        nn.Linear(28*28,300),
        nn.ReLU(),
        nn.Linear(300,10))

  def forward(self, x):
    x = self.flatten(x)
    return self.model(x)

def train_MLP(dataloader, model, loss_fn, optimizer):
  model.train()
  for batch, (samples, labels) in enumerate(dataloader):
    samples, labels = samples.to(DEVICE), labels.to(DEVICE)
    # forward pass
    y_tilde = model(samples)
    loss = loss_fn(y_tilde, labels)
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print stats
    if batch % 100 == 0:
      sys.stdout.write("\x1b[0G" + f"batch {batch} of {len(dataloader)} -->  loss: {loss.item():>7f}")
      sys.stdout.flush()

def evaluate_MLP(dataloader, model, loss_fn):
  loss_sum, n_correct = 0, 0
  with torch.no_grad():
    for samples, labels in dataloader:
      samples, labels = samples.to(DEVICE), labels.to(DEVICE)
      y_tilde = model(samples)
      loss_sum += loss_fn(y_tilde, labels).item() 
      n_correct += (y_tilde.argmax(1) == labels).type(torch.float).sum().item()
  loss_sum /= len(dataloader)
  n_correct /= len(dataloader.dataset)
  print(f"\nEvaluation Error: \n  Accuracy -> {(100 * n_correct):>0.1f}%, Avg Loss -> {loss_sum:>8f}\n")
  return 100 * n_correct

def load_data():
  # Download training data from open datasets.
  training_data = datasets.MNIST(
      root="../dataset",
      train=True,
      download=True,
      transform=ToTensor())
  training_size = int(0.8 * len(training_data))
  validation_size = len(training_data) - training_size
  training_data, validation_data = torch.utils.data.random_split(training_data, 
      [training_size, validation_size])
  # Download test data from open datasets.
  test_data = datasets.MNIST(
      root="../dataset",
      train=False,
      download=True,
      transform=ToTensor())
  return training_data, validation_data, test_data

if __name__ == "__main__":
  load = True
  if len(sys.argv) < 2:
    load = False
  training_data, validation_data, test_data = load_data()
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
  model = MLP().to(DEVICE)
  loss_fn = nn.CrossEntropyLoss()
  print(model)
  if load:
    model.load_state_dict(torch.load(sys.argv[1]))
  else:
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    best_accuracy, count = 0, 0
    for t in range(EPOCHS):
      print(f"************ Epoch {t} ************\n")
      train_MLP(train_dataloader, model, loss_fn, optimizer)
      accuracy = evaluate_MLP(validation_dataloader, model, loss_fn)
      count = count + 1 if accuracy <= best_accuracy else 0
      best_accuracy = max(accuracy, best_accuracy)
      if (count > 3):
        break
  accuracy = evaluate_MLP(test_dataloader, model, loss_fn)
  print(f"\nTest set accuracy of trained model: {accuracy}")
  torch.save(model.state_dict(), "model_10-31-1339.pth")
  print("DONE")
