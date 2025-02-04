import torch
import torch.nn as nn
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def plot_latent_space_with_labels(num_classes, data_loader, model, device):
    d = {i: [] for i in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            embedding = model.encoder(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()

def corrupt(x: torch.Tensor, mean: float = 0,
            var: float = 1, alpha: float = 0.1,
            device: str = "cpu") -> torch.Tensor:
    """
        Add Gaussian noise to given image, in the
        given device. Returns the corrupted image
        tensor in the given device.

        Args:
            x (Tensor): The image(s) as a torch Tensor to be corrupted.

            mean (float): Mean of the Gaussian Noise.

            var (float): Variance of the Gaussian Noise.

            alpha (float): Magnitude of the Gaussian Noise.

            device (str): Device to operate on.

        Returns:
              Tensor: The corrupted image(s).
    """
    x = x.to(device)
    noise = alpha * (torch.randn(x.size()).to(device) * var + mean)
    return x + noise

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class LatentLayer(nn.Module):

    """
        This is a latent layer for any VAE application, for the
        purpose of wrapping the reparameterization trick and ladder
        application.

        When initializing, a module subclass and initialization
        arguments are given. LatentLayer, initializes given layer
        and automatically reparameterizes the output of said layer
        when forward pass is called.

        The main network, mu-predictor network and logvar-predictor
        networks are given initialized by the user. The layer manages
        the passes.

        Algorithm:
            2 LatentLayer objects are (should) be coupled with their
            "latent" argument. They both have a control over this
            shared data store.

            When the first LatentLayer in the network suffers a forward
            pass, it internally calculates the latent distribution and
            saves it to the shared latent data store.

            When the second LatentLayer, the one that is coupled with
            the one from above, in the network suffers a forward pass,
            it realizes that there is already data in the shared latent
            data store. It calculates its own latent distribution first,
            then calculates the common distribution with the one that is
            in the shared latent data store. The common distribution is
            registered to the shared latent data store. This registration
            overrides the previous data.

            Now, after the full forward pass of the hosting model ends, the
            host model can use the information within the shared latent data
            store. There is the common distribution, to be used within KL
            divergence loss.

            Do not forget to flush any instance of this class, after a full
            forward pass and backward pass of the host model. Otherwise, the
            first LatentLayer in the hierarchy will look at the shared latent
            data store, and see that there is already data in there. So it will
            act like a decoder layer on its own, even though it is (probably)
            an encoder layer. A faulty "common" distribution than would be
            saved in the shared store.

        Args:
            network: The main network to calculate the forward pass. Must be
                given initialized.

            mu_network: The network to calculate the mean of the supposed latent
                distribution. Must be given initialized.

            logvar_network: The network to calculate the log-var of the supposed latent
                distribution. Must be given initialized.

            activation: The activation function to be used after the forward pass of
                the main network.

            latent (list): A list of 2 elements; first for mean, second for logvar.
                This should be given initialized from a hosting torch module, because
                this is the shared latent data store.

            is_ladder (bool): If this is true, the LatentLayer instance acts like as
                if it is within a LadderVAE model. Otherwise, it just behaves as normal.
                It does not use the shared latent data space even if given any. Just
                applies the reparameterizaiton trick. The default is True.

    """

    def __init__(self,
                 network: nn.Module,
                 mu_network: nn.Module,
                 logvar_network: nn.Module,
                 activation: nn.Module,
                 latent: list = None,
                 is_ladder: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.mu_network = mu_network
        self.logvar_network = logvar_network
        self.activation = activation
        self.is_ladder = is_ladder

        # If this layer is within a decoder, self.zs will containt the
        # common weighted distribution of encoder and decoder.
        #
        # If this layer is within an encoder, self.zs will just contain
        # the normally predicted mean and logvar from the encoder
        self.zs = latent if latent is not None else [None, None]
        self.mu = None
        self.logvar = None

    def forward(self, x):
        out = self.activation(self.network(x))

        mu = self.mu_network(out)
        logvar = self.logvar_network(out)

        if not self.training:
            return self.reparameterize(mu, logvar)

        if self.is_filled() and self.is_ladder:
            encoder_var_ = torch.exp(self.zs[1])
            decoder_var_ = logvar.exp()

            common_sigma = 1 / (encoder_var_ + decoder_var_)
            common_mu = (self.zs[0] / encoder_var_ + mu / decoder_var_) * common_sigma
            common_logvar = torch.log(torch.pow(common_sigma, 2))
            self.register(common_mu, common_logvar)
            return self.reparameterize(mu, logvar)
        else:
            # We still register even if this is not a Ladder implementation,
            # So later on we can perform kld loss on them
            self.mu = mu
            self.logvar = logvar
            self.register(mu, logvar)
            return out

    def get_latent_distribution(self, x) -> tuple:
        """
            Get the mean and logvar of the latent distribution predicted
            by this layer.

            Returns:
                tuple: mu, logvar
        """
        out = self.activation(self.network(x))
        return self.mu_network(out), self.logvar_network(out)

    def flush(self):
        """
            Flush the registered information about the predicted distributions.

            This is to clean the memory. This may not work on its own for non-arm
            devices. Arm system, specifically Metal Performance Shaders, manage
            its own memory clean up. So just marking a variable with "del" as
            deleted, would make MPS system garbage collect them.

            However, on CUDA, you on your own have to call "empty_cache".
        """
        obj1, obj2 = self.zs[0], self.zs[1]
        del obj1, obj2
        self.zs[0] = None
        self.zs[1] = None

    def register(self, mu, logvar):
        """
            Register the distribution to the shared latent data store.

            Args:

                mu: Mean of the predicted distribution.

                logvar: The log-variance of the predicted distribution.
        """
        self.zs[0] = mu
        self.zs[1] = logvar

    def is_filled(self) -> bool:
        """
            Checks if the shared latent data store is filled.
        """
        return (self.zs[0] is not None) and (self.zs[1] is not None)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class LadderVAE(nn.Module):

    """
        The Ladder VAE implementation for MNIST dataset.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = nn.LeakyReLU(0.2)

        self.upper_latent_network_decoder = nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.upper_latent_network_encoder = nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=0)
        self.upper_mu_network = nn.Conv2d(64, 64, stride=1, kernel_size=1)
        self.upper_logvar_network = nn.Conv2d(64, 64, stride=1, kernel_size=1)

        self.lower_latent_network_encoder = nn.Linear(3136, 2)
        self.lower_mu_network = nn.Linear(2, 2)
        self.lower_logvar_network = nn.Linear(2, 2)

        self.shared_latent = [None, None]
        self.core_latent = [None, None]

        self.upper_latent_layer_encoder = LatentLayer(
            self.upper_latent_network_encoder,
            nn.Conv2d(64, 64, stride=1, kernel_size=1),
            nn.Conv2d(64, 64, stride=1, kernel_size=1),
            self.activation,
            self.shared_latent
        )

        self.upper_latent_layer_decoder = LatentLayer(
            self.upper_latent_network_decoder,
            nn.Conv2d(64, 64, stride=1, kernel_size=1),
            nn.Conv2d(64, 64, stride=1, kernel_size=1),
            self.activation,
            self.shared_latent
        )

        self.lower_latent_layer_encoder = LatentLayer(
            self.lower_latent_network_encoder,
            self.lower_mu_network,
            self.lower_logvar_network,
            self.activation,
            self.core_latent,
            False
        )

        self.encoder = nn.Sequential(  # 784
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            self.activation,
            self.upper_latent_layer_encoder,
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            self.activation,
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            self.activation,
            nn.Flatten(),
            self.lower_latent_layer_encoder
            # (64, 7, 7)
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            self.activation,
            self.upper_latent_layer_decoder,
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            self.activation,
            nn.Conv2d(32, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            self.activation,
            nn.ConvTranspose2d(32, 16, stride=(1, 1), kernel_size=(2, 2), padding=0),
            self.activation,
            nn.Conv2d(16, 1, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))

        loss_ = self.kld_loss(*self.core_latent) if self.training else 0
        loss_ += self.kld_loss_multiple(
            *self.shared_latent,
            self.upper_latent_layer_encoder.mu,
            self.upper_latent_layer_encoder.logvar
        ) if self.training else 0
        return out, loss_

    def kld_loss(self, mu, logvar):
        """
            Calculate basic KL divergence from normal Gaussian
            distribution.

            Args:
                mu: Mean of the source distribution.

                logvar: Log of variance of the source distribution.
        """
        return -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))

    def kld_loss_multiple(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.exp()
        sigma2 = logvar2.exp()
        return torch.sum(torch.log(torch.sqrt(sigma2 / sigma1)) + (sigma1 + (mu1 - mu2) ** 2) / (2 * sigma2) - 0.5)

    def recon_loss(self, recon_x, x, *args, **kwargs):
        """
            The reconstruction loss. Any arguments other than specified,
            are passed to the torch.norm function. This is to control the
            norm that is used as the distance metric.

            Args:
                recon_x: The reconstructed output.

                x: The input to the network, that is to be reconstructed.
        """
        diff = torch.abs(recon_x - x)
        return torch.mean(torch.norm(diff, *args, **kwargs))

    def flush(self):
        """
            Flush the registered information about the predicted distributions.

            This is to clean the memory. This may not work on its own for non-arm
            devices. Arm system, specifically Metal Performance Shaders, manage
            its own memory clean up. So just marking a variable with "del" as
            deleted, would make MPS system garbage collect them.

            However, on CUDA, you on your own have to call "empty_cache".
        """
        obj1, obj2 = self.shared_latent[0], self.shared_latent[1]
        del obj1, obj2
        self.shared_latent[0] = None
        self.shared_latent[1] = None
        self.lower_latent_layer_encoder.flush()



