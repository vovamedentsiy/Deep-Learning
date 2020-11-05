import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        input_dim = 128
        num_hidden1 = 256
        num_hidden2 = 512
        num_hidden3 = 1024

        self.layer0 = nn.Linear(latent_dim, input_dim)
        self.activ0 = nn.LeakyReLU(0.2, inplace=True)

        self.layer1 = nn.Linear(input_dim, num_hidden1)
        self.activ1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm1d(num_hidden1)

        self.layer2 = nn.Linear(num_hidden1, num_hidden2)
        self.activ2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm1d(num_hidden2)

        self.layer3 = nn.Linear(num_hidden2, num_hidden3)
        self.activ3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm1d(num_hidden3)

        self.layer4 = nn.Linear(num_hidden3, 784)
        self.activ4 = nn.Tanh()

    def forward(self, z):
        # Generate images from z
        z = z.to('cuda:0')

        z_ = self.layer0(z)
        z_ = self.activ0(z_)

        z_ = self.layer1(z_)
        z_ = self.activ1(z_)
        z_ = self.bn1(z_)
        z_ = self.layer2(z_)
        z_ = self.activ2(z_)
        z_ = self.bn2(z_)
        z_ = self.layer3(z_)
        z_ = self.activ3(z_)
        z_ = self.bn3(z_)
        z_ = self.layer4(z_)
        z_ = self.activ4(z_)


        return z_


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        input_dim = 784
        num_hidden1 = 512
        num_hidden2 = 256
        num_hidden3 = 1

        self.layer1 = nn.Linear(input_dim, num_hidden1)
        self.activ1 = nn.LeakyReLU(0.2, inplace=True)
        self.dr1 = nn.Dropout(p=0.4)

        self.layer2 = nn.Linear(num_hidden1, num_hidden2)
        self.activ2 = nn.LeakyReLU(0.2, inplace=True)
        self.dr2 = nn.Dropout(p=0.4)

        self.layer3 = nn.Linear(num_hidden2, num_hidden3)
        self.activ3 = nn.Sigmoid()


    def forward(self, img):

        img = img.to('cuda:0')
        # return discriminator score for img
        z = self.layer1(img)
        z = self.activ1(z)
        z = self.dr1(z)

        z = self.layer2(z)
        z = self.activ2(z)
        z = self.dr2(z)

        z = self.layer3(z)
        z = self.activ3(z)

        return z


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):

    loss = torch.nn.BCELoss()
    loss_gen_list = []
    loss_discr_list = []

    for epoch in range(args.n_epochs):

        loss_gen_ = []
        loss_discr_ = []

        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()

            labels1 = torch.FloatTensor(torch.ones([imgs.size(0), 1], requires_grad=False)).to('cuda:0')
            labels0 = torch.FloatTensor(torch.zeros([imgs.size(0), 1], requires_grad=False)).to('cuda:0')
            z = np.random.normal(0, 1, (imgs.shape[0], latent_dim))
            z = torch.from_numpy(z).float().to('cuda:0')
            imgs_vector = imgs.reshape(imgs.size(0), 784)

            # Train Generator
            # ---------------

            gen_imgs = generator(z)
            gen_loss = loss(discriminator(gen_imgs), labels1)

            optimizer_G.zero_grad()

            gen_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------

            if np.random.binomial(n=1, p=0.2) != 1:

              discr_loss1 = loss(discriminator(imgs_vector), labels1)
              discr_loss2 = loss(discriminator(gen_imgs.detach()), labels0)
              discr_loss = (discr_loss1 + discr_loss2) #/ 2

              optimizer_D.zero_grad()
              discr_loss.backward()
              optimizer_D.step()

            # Save Images
            # -----------
            loss_gen_.append(gen_loss.item())
            loss_discr_.append(discr_loss.item())


            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                generator.eval()
                gen_imgs = generator(z)

                gen_imgs = gen_imgs.reshape(gen_imgs.size(0), 1, 28, 28)
                save_image(gen_imgs[:25], 'images/{}.png'.format(batches_done), nrow=5, normalize=True)
                generator.train()

        loss_gen_list.append(np.mean(loss_gen_))
        loss_discr_list.append(np.mean(loss_discr_))
        print('Epoch: ', epoch, ' G loss: ', np.mean(loss_gen_), ' D loss: ', np.mean(loss_discr_))

        if epoch%4 == 0 or epoch == 199:

          name_model = 'all_models/model_'+str(int(epoch))+'.pickle'
          torch.save(generator.state_dict(), name_model)

    np.savetxt('loss_gen.txt', np.array(loss_gen_list))
    np.savetxt('loss_discr.txt', np.array(loss_discr_list))


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    os.makedirs('all_models', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")

    # already saved during training
    '''
    for generating images
    model = Generator().to('cuda:0')
    model.load_state_dict(torch.load("mnist_generator.pt"))
    model.eval()
    z = np.random.normal(0, 1, (2, 100))
    z = torch.from_numpy(z).float()
    t = np.linspace(z[0], z[1], 9)
    t = torch.from_numpy(t).float().to('cuda:0')

    gen_imgs = model(t)
    gen_imgs = gen_imgs.reshape(gen_imgs.size(0), 1, 28, 28)
    save_image(gen_imgs, 'generated_transform.png', nrow=9, normalize=True)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
