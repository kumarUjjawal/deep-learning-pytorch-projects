"""
We'll use GAN to generate photorealistic images. We will be using DCGAN introduced in [Radford et al., 2015].

Refrences:
    GAN [Goodfellow et. al., 2014]
    DCGAN [Radford et. al., 2016]
    Dive into Deep Learning [Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola, 2020]
    Pokemondb [https://pokemondb.net/sprites]
"""
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings

# The Dataset

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip', 'c065c0e2593b8b161a2d7873e42418bf6a21106c')
data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)

batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5,0.5)
])

pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size, shuffle=True,
    num_workers=d2l.get_dataloader_workers())

# The Generator

class GBlock(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, **kwargs):
        super(GBlock, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_chnnels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv2d_trans(x)))
n_G = 64
net_G = nn.Sequential(
    GBlock(in_channels=100, out_channels=n_G*8, strides=1, padding=0),
    GBlock(in_channels=n_G*8, out_channels=n_G*4),
    GBlock(in_channels=n_G*4, out_channels=n_G*2),
    GBlock(in_channels=n_G*2, out_channels=n_G),
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3, kernel_size=4,
                       strides=2, padding=1, bias=False),
    nn.Tanh())

# The Discriminator

class DBlock(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, alpha=0.2, **kwargs):
        super(DBlock, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv2d(x)))

n_D = 64
net_D = nn.Sequential(
    DBlock(in_channels=n_D, out_channels=n_D*2),
    DBlock(in_channels=n_D*2, out_channels=n_D*4),
    DBlock(in_channels=n_D*4, out_channels=n_D*8),
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))

# Training

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5, 0.999]}
    trainer__D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2, figsize=(5,5), legend=['discriminator', 'generator'])

    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer__D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        fake_x = net_G(Z).permute(0, 2, 3, 1)/ 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
            for i in range(len(fake_x)//7)], dim=0)

        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)

        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f},'
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')

latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)


