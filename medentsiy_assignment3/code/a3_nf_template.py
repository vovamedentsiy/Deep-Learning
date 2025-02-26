import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid




def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    #m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(k), torch.eye(k))
    #logp = m.log_prob(x)

    logp = (-0.5*x.pow(2)-torch.log(torch.sqrt(torch.tensor(2*np.pi)))).sum(dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = torch.randn(size)

    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(in_features=c_in, out_features=n_hidden, bias=True), nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True), nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=2 * c_in, bias=True)
        )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        if not reverse:
            z_ = self.mask * z
            s, t = torch.chunk(self.nn(z_), 2, 1)
            s = torch.tanh(s)

            ldj += ((1 - self.mask)*s).sum(dim=1)

            z = z_ + (1 - self.mask) * (z * torch.exp(s) + t)
        else:
            z_ = self.mask * z
            s, t = torch.chunk(self.nn(z_), 2, 1)
            s = torch.tanh(s)

            z = z_ + (1 - self.mask) * (z - t) * torch.exp(-s)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example

        log_pz = log_prior(z)
        log_px = (log_pz + ldj).mean()

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        #raise NotImplementedError
        z, ldj = self.flow(z, ldj, reverse = True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    loss_bpd = 0

    for i, (d, l) in enumerate(data, 1):

        loss = -model(d.to('cuda:0'))

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 3)
            optimizer.step()

        loss_bpd += loss.item()

    loss_bpd = loss_bpd/i
    avg_bpd = (loss_bpd ) / (784 * np.log(2.))

    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------

        sampled_ims = model.sample(25).detach()
        sampled_ims = sampled_ims.reshape(25, 1, 28, 28).cpu()
        grid_img = make_grid(sampled_ims, nrow=5, normalize=True).detach()

        plt.title('Sampled Images')
        plt.imshow(grid_img.permute(1, 2, 0), )  # , (1,2,0)))
        title = 'sampled_img_nvp'+str(epoch)+'.png'
        plt.savefig(title, dpi=100)


    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')

    name_model = 'real_nvp.pickle'
    torch.save(model.state_dict(), name_model)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()

    main()
