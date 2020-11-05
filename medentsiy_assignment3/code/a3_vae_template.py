import argparse

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
import numpy

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.layerH = nn.Linear(784, hidden_dim, bias = True)
        self.layerHActivation = nn.Tanh()
        self.layerMean = nn.Linear(hidden_dim, z_dim, bias = True)
        self.layerStd = nn.Linear(hidden_dim, z_dim, bias = True)
        self.layerStdActivation = nn.Softplus()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        # Make sure that any constraints are enforced.


        h = self.layerH(input)
        h = self.layerHActivation(h)

        mean, std = self.layerMean(h), self.layerStdActivation(self.layerStd(h))

        #raise NotImplementedError()

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.layerH1 = nn.Linear(z_dim, hidden_dim, bias=True)
        self.layerH1Activation = nn.Tanh()

        self.layerH2 = nn.Linear(hidden_dim, 784, bias=True)
        self.layerH2Activation = nn.Sigmoid()




    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        h = self.layerH1(input)
        h = self.layerH1Activation(h)

        mean = self.layerH2Activation(self.layerH2(h))

        #raise NotImplementedError()

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        b_size = input.size(0)

        meanEnc, sigmaEnc = self.encoder(input)
        randSample = torch.randn(size=(b_size, self.z_dim))
        inputZ = torch.mul(sigmaEnc, randSample)
        inputZ = inputZ + meanEnc
        meanDec = self.decoder(inputZ)

        p = torch.sum(input * torch.log(meanDec) + (1 - input) * torch.log(1 - meanDec), dim=-1)
        lossRecon = -torch.mean(p)
        lossReg = -1/2*torch.sum(1+torch.log(sigmaEnc ** 2) - meanEnc ** 2 - sigmaEnc ** 2, dim = -1)
        lossReg = torch.mean(lossReg)
        average_negative_elbo = lossRecon + lossReg
        #raise NotImplementedError()


        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        randZ = torch.randn(size=(n_samples, self.z_dim))
        im_means = self.decoder(randZ)
        sampled_ims = torch.bernoulli(im_means)

        #sampled_ims, im_means = None, None
        #raise NotImplementedError()

        return sampled_ims, im_means


def plot_manifold(load_model = True, name_of_model= '2dim_model_vae.pickle'):
    '''
    to answer question 1.15
    '''
    if load_model:
        model = VAE(z_dim=2)
        model.load_state_dict(torch.load(name_of_model))
        model.eval()
    else:
        model.eval()

    import numpy as np
    from scipy.stats import norm

    p = np.linspace(0.01, 0.99, 20)
    z = norm.ppf(p)
    z_code = []
    for z1 in z:
        for z2 in z:
            z_code.append([z1, z2])
    z_code = torch.FloatTensor(z_code)

    im_means = model.decoder(z_code).reshape(400, 1, 28, 28)

    plt.figure()
    grid_img = make_grid(im_means, nrow=20).detach()
    plt.title('Learned MNIST manifold')
    plt.imshow(grid_img.permute(1, 2, 0), )

    plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)

    title = 'vae_dim2.png'
    plt.savefig(title, dpi=100)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = []
    if model.training:
        for step, batch in enumerate(data):
            # print(batch)
            s = batch.size()
            batch = batch.reshape(s[0], s[1] * s[2] * s[3])
            elbo = model(batch)
            average_epoch_elbo.append(float(elbo))
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()
    else:
        for step, batch in enumerate(data):
            s = batch.size()
            batch = batch.reshape(s[0], s[1] * s[2] * s[3])
            elbo = model(batch)
            average_epoch_elbo.append(float(elbo))

    average_epoch_elbo = numpy.mean(average_epoch_elbo)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():


    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    ''' plot before training '''
    sampled_ims, im_means = model.sample(100)
    im_means = im_means.reshape(100, 1, 28, 28)
    sampled_ims = sampled_ims.reshape(100, 1, 28, 28)

    grid_img = make_grid(sampled_ims, nrow=10).detach()
    plt.title('Sampled. Before training')
    plt.imshow(grid_img.permute(1, 2, 0), )  # , (1,2,0)))
    title = 'sampled0.png'
    plt.savefig(title, dpi=100)

    grid_img = make_grid(im_means, nrow=10).detach()
    plt.title('Mean. Before training')
    plt.imshow(grid_img.permute(1, 2, 0), )  # , (1,2,0)))
    title = 'mean0.png'
    plt.savefig(title, dpi=100)


    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        sampled_ims, im_means = model.sample(100)
        im_means = im_means.reshape(100, 1, 28, 28)
        sampled_ims = sampled_ims.reshape(100, 1, 28, 28)

        grid_img = make_grid(sampled_ims, nrow=10).detach()
        plt.title('Sampled. Epoch '+str(epoch+1))
        plt.imshow(grid_img.permute(1, 2, 0), )  # , (1,2,0)))
        title = 'sampled'+str(epoch+1)+'.png'
        plt.savefig(title, dpi=100)

        grid_img = make_grid(im_means, nrow=10).detach()
        plt.title('Mean. Epoch '+str(epoch+1))
        plt.imshow(grid_img.permute(1, 2, 0), )  # , (1,2,0)))
        title = 'mean'+str(epoch+1)+'.png'
        plt.savefig(title, dpi=100)


    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

    name_model = 'model_vae.pickle'
    torch.save(model.state_dict(), name_model)


    if ARGS.zdim == 2:
        plot_manifold(False)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')


    ARGS = parser.parse_args()

    main()
