import torch
import torchvision.datasets as datasets
from tqdm import tqdm # progressbar
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms # augmentations
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
# loss weighing 
alpha = 1 
beta = 1


# dataset stuff
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# model, optim and loss
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss(reduction="sum")

# traning loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM) # reshape makes a copy, so view is used?
        x_hat, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_hat, x) # pushing it towards a std gaussian
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # ?

        # backpropagation
        loss = alpha*reconstruction_loss + beta*kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


model = model.to("cpu")
def inference(digit, num_examples=1):
    """  bla  """
    images = []
    idx = 0
    for x, y in dataset: 
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
    
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784)) # get mu and sigma from model
        encodings_digit.append((mu, sigma)) # store them with their digit

    mu, sigma = encodings_digit[digit] 
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon # reparameterize
        y = model.decode(z)
        y = y.view(-1, 1, 28, 28)
        save_image(y, f"generated_vae_{digit}_mnist_{example}.png")

for idx in range(10):
    inference(idx, num_examples=5) 
        





