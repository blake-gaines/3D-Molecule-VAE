## parts taken from https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
## and https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grid_loader import setup_data_loaders
import os 
from tqdm import tqdm
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal

# torch.autograd.set_detect_anomaly(True)

RESOLUTION = 32
SMALLEST_DIM = 6
IMG_CHANNELS = 5
HIDDEN_DIM = (SMALLEST_DIM**3)*64
NUM_FEATURES = IMG_CHANNELS*(RESOLUTION**3)

print("HIDDEN_DIM", HIDDEN_DIM)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape


    def forward(self, input):
        return input.view(input.size(0), *self.output_shape)
        # return input.view(input.size(0), size, 6, 6, 6)


# q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            Flatten(),
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(z_dim, z_dim)
        self.fc22 = nn.Linear(z_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv_layers(x)
        mu = self.fc21(x)
        logVar = self.fc22(x)
        return mu, logVar


# p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_channels):
        super().__init__()
        # setup the two linear transformations used
        self.deconv_layers = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),            
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            UnFlatten(output_shape=(64, SMALLEST_DIM, SMALLEST_DIM, SMALLEST_DIM)),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=image_channels, kernel_size=3, stride=1, padding=0), # dimensions should be as original
            nn.Sigmoid(),
        )

    def forward(self, z):
        loc_img = self.deconv_layers(z)
        return loc_img


class VAE(nn.Module):
    def __init__(self, n_channels, z_dim, hidden_dim, beta=1, use_cuda=False):
        super().__init__()

        self.encoder = Encoder(z_dim, hidden_dim, image_channels=n_channels)
        self.decoder = Decoder(z_dim, hidden_dim, image_channels=n_channels)

        if use_cuda:
            self.cuda()

        self.n_channels = n_channels
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.beta = beta

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, data):
        mu, logVar = self.encoder(data)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

    def compute_loss(self, truth, prediction, mu, logVar):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        reconstruction_loss = F.binary_cross_entropy(prediction, truth, reduction="sum")
        loss = reconstruction_loss + self.beta*kl_divergence
        return loss, kl_divergence, reconstruction_loss

    def sample_from_latent_space(self, num_generations, grid_to_molecule):
        sampled_latent_vectors = MultivariateNormal(torch.zeros(self.z_dim), scale_tril=torch.eye(self.z_dim)).sample((num_generations,)).double().cuda()
        sampled_generations = list(self.decoder(sampled_latent_vectors).cpu().detach().numpy())

        filenames = []
        for i, generation in enumerate(sampled_generations):
            new_mol = grid_to_molecule(generation)
            if isinstance(new_mol, str):
                print(new_mol)
                continue
            new_mol_smile = new_mol.write("can").split('\t')[0]
            filename = f"./images/sampled_generation_{i}.jpg"
            filenames.append(filename)
            new_mol.draw(show=False, filename=filename)
        return sampled_generations, filenames


def main(args):
    directory = "./dsgdb9nsd.xyz/"
    train_loader, test_loader = setup_data_loaders(
        directory, os.listdir(directory), use_cuda=args.cuda, batch_size=args.batch_size, resolution=RESOLUTION, shuffle=True #UNFINISHED: SHUFFLE_TRUE
    )

    print("CUDA:", args.cuda)
    print("WANDB:", args.wandb)

    vae = VAE(n_channels=5, z_dim=256, hidden_dim=HIDDEN_DIM, use_cuda=args.cuda)
    vae.double()
    print("MODEL:", vae)

    adam_args = {"lr": args.learning_rate}
    optimizer = torch.optim.Adam(vae.parameters(), **adam_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    start_epoch = 0
    if args.checkpoint is not None:
        print("Loading From Checkpoint: ", args.checkpoint)
        state_dict = torch.load(args.checkpoint)
        vae.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        start_epoch = state_dict["epoch"]+1
        if args.wandb: wandb.init(config=args, project="molecule_vae", resume=True)
    else:
        if args.wandb: wandb.init(config=args, project="molecule_vae")

    if args.wandb: 
        wandb.watch(vae)
        wandb.save("./vae.py", policy="now")

    train_loss = []
    test_loss = []

    for epoch in range(start_epoch, args.num_epochs):
        vae.train()

        epoch_loss = 0.0

        pbar = tqdm(train_loader)
        for x, _ in pbar:
            if args.cuda:
                x = x.cuda()

            prediction, mu, logVar = vae(x)
            loss, kl_divergence, reconstruction_loss = vae.compute_loss(x, prediction, mu, logVar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if args.wandb: 
                wandb.log({
                    "Train Batch Loss": loss,
                    "Train Batch KL loss": kl_divergence.item(),
                    "Train Batch Reconstruction Loss": reconstruction_loss.item()
                    })

            pbar.set_postfix_str(str(round(loss, 2)))
            epoch_loss += loss

        average_epoch_loss_train = args.batch_size*epoch_loss / len(train_loader.dataset)
        train_loss.append(average_epoch_loss_train)

        scheduler.step()

        if epoch % args.test_frequency != 0:
            if args.wandb: wandb.log({"Average Training Batch Loss": average_epoch_loss_train})
        else:
            save_name = "model.pt"
            torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_epoch_loss_train,
            }, "./saves/" + save_name)
            if args.wandb: wandb.save("./saves/"+save_name, policy="now")
            vae.eval()

            epoch_test_loss = 0.0
            with torch.no_grad():
                for i, xy in enumerate(test_loader):
                    x, _ = xy

                    if args.cuda:
                        x = x.cuda()
                    
                    out, mu, logVar = vae(x)

                    try:
                        if args.wandb and i == 0:
                            x_cpu = x.cpu().detach().numpy()
                            out_cpu = out.cpu().detach().numpy()
                            old_image_list, new_image_list = [], []
                            for j in range(min(x_cpu.shape[0], 10)):
                                old_mol = test_loader.dataset.grid_to_molecule(x_cpu[j])
                                new_mol = test_loader.dataset.grid_to_molecule(out_cpu[j])
                                if type(old_mol) == str or type(new_mol) == str: 
                                    continue
                                old_img_path = f"./images/test_old_{j}.jpg"
                                new_img_path = f"./images/test_new_{j}.jpg"
                                old_mol.draw(show=False, filename=old_img_path)
                                new_mol.draw(show=False, filename=new_img_path)
                                old_image_list.append(wandb.Image(old_img_path))
                                new_image_list.append(wandb.Image(new_img_path))
                            if old_image_list and new_image_list:
                                wandb.log({
                                    f"Test Original": old_image_list,
                                    f"Test Reconstruction": new_image_list
                                }, commit=True)
                            _, filenames = vae.sample_from_latent_space(5, test_loader.dataset.grid_to_molecule)
                            if filenames:
                                wandb.log({"Sampled Generations:": [wandb.Image(filename) for filename in filenames]})
                    except Exception as e:
                        print("Failed logging pictures:", e)
                    loss, kl_divergence, reconstruction_loss = vae.compute_loss(x, out, mu, logVar)

                    epoch_test_loss += loss.item()

            average_epoch_loss_test = args.batch_size*epoch_test_loss / i
            test_loss.append(average_epoch_loss_test)
            if args.wandb: wandb.log({"Average Training Batch Loss": average_epoch_loss_train, "Average Testing Batch Loss": average_epoch_loss_test})

    return vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=40, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-tf",
        "--test-frequency",
        default=1,
        type=int,
        help="how often we evaluate the test set",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-5, type=float, help="learning rate"
    )
    parser.add_argument(
        "-bs", "--batch-size", default=32, type=int, help="batch size"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False, help="whether to track model"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to checkpoint"
    )

    args = parser.parse_args()

    model = main(args)