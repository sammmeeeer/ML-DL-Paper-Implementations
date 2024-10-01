import os 
import json 
import argparse 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets 
from tensorboardX import SummaryWriter


# Preprocessing 
# Preprocessing 
def preprocess_mnist(split='train'):
    data = datasets.MNIST('./data', train=split=='train', download=True)
    n = 7291 if split == 'train' else 2007 
    rp = np.random.permutation(len(data))[:n]
    X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)
    Y = torch.full((n, 10), -1.0, dtype=torch.float32)
    for i, ix in enumerate(rp):
        I, yint = data[int(ix)]
        xi = torch.from_numpy(np.array(I, dtype=np.float32)) / 127.5 - 1.0 
        xi = xi[None, None, ...]
        xi = F.interpolate(xi, (16, 16), mode='bilinear')
        X[i] = xi[0]
        Y[i, yint] = 1.0  
    return X, Y  

# Architecture 
class Net(nn.Module):
    def __init__(self):
        super().__init__()


        winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5
        self.macs = 0 
        self.acts = 0 

        # H1 Layer 
        self.H1w = nn.Parameter(winit(5*5*1, 12, 1, 5, 5))
        self.H1b = nn.Parameter(torch.zeros(12, 8, 8))
        self.macs += (5*5*1) * (8*8) * 12 
        self.acts += (8*8) * 12 

        # H2 Layer 
        self.H2w = nn.Parameter(winit(5*5*8, 12, 8, 5, 5))
        self.H2b = nn.Parameter(torch.zeros(12, 4, 4))
        self.macs += (5*5*8) * (4*4) * 12 
        self.acts += (4*4) * 12 

        # H3 Layer 
        self.H3w = nn.Parameter(winit(4*4*12, 4*4*12, 30))
        self.H3b = nn.Parameter(torch.zeros(30))
        self.macs += (4*4*12) * 30 
        self.acts += 30 

        # Output Layer 
        self.outw = nn.Parameter(winit(30, 30, 10))
        self.outb = nn.Parameter(torch.zeros(10))
        self.macs += 30 * 10 
        self.acts += 10 

    def forward(self, x):
        if self.training:
            shift_x, shift_y = np.random.randint(-1, 2, size=2)
            x = torch.roll(x, (shift_x, shift_y), (2, 3))

        # H1 layer
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = F.relu(x)

        # H2 layer
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0)
        slice1 = F.conv2d(x[:, 0:8], self.H2w[0:4], stride=2)
        slice2 = F.conv2d(x[:, 4:12], self.H2w[4:8], stride=2)
        slice3 = F.conv2d(torch.cat((x[:, 0:4], x[:, 8:12]), dim=1), self.H2w[8:12], stride=2)
        x = torch.cat((slice1, slice2, slice3), dim=1) + self.H2b
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        # H3 layer
        x = x.flatten(start_dim=1)
        x = x @ self.H3w + self.H3b
        x = F.relu(x)

        # Output layer
        x = x @ self.outw + self.outb
        return x


def main():
    parser = argparse.ArgumentParser(description="Train a modernized 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--output-dir', '-o', type=str, default='out/modern', help="output directory for training logs")
    parser.add_argument('--epochs', '-e', type=int, default=23, help="Number of epochs to train")
    args = parser.parse_args()
    print(vars(args))

    # Init RNG
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # Preprocess and load data
    Xtr, Ytr = preprocess_mnist('train')
    Xte, Yte = preprocess_mnist('test')

    # Init model
    model = Net()
    print("model stats:")
    print("# params:      ", sum(p.numel() for p in model.parameters()))
    print("# MACs:        ", model.macs)
    print("# activations: ", model.acts)

    # Init optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        model.eval()
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        with torch.no_grad():
            Yhat = model(X)
            loss = F.cross_entropy(Yhat, Y.argmax(dim=1))
            err = (Y.argmax(dim=1) != Yhat.argmax(dim=1)).float().mean()
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
        writer.add_scalar(f'error/{split}', err.item()*100, epoch)
        writer.add_scalar(f'loss/{split}', loss.item(), epoch)

    # Train
    for epoch in range(args.epochs):
        # Learning rate decay
        alpha = epoch / (args.epochs - 1)
        for g in optimizer.param_groups:
            g['lr'] = (1 - alpha) * args.learning_rate + alpha * (args.learning_rate / 3)

        model.train()
        for step in range(Xtr.size(0)):
            x, y = Xtr[step:step+1], Ytr[step:step+1]  # Select a single sample
            yhat = model(x)
            loss = F.cross_entropy(yhat, y.argmax(dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}")
        eval_split('train')
        eval_split('test')

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))

if __name__ == '__main__':
    main()
