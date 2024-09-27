import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from network_ad.config import LOGS_DIR, MAX_PLOT_POINTS, BINARY_CLASS_NAMES, MULTIClASS_CLASS_NAMES, VAL_RATIO
from network_ad.unsupervised.autoencoder_datamodule import AutoencoderDataModule


class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32, learning_rate=1e-3, dropout_rate=0.1,
                 max_training_steps=None):
        """
        :param input_dim: The input dimension (number of features) of the autoencoder
        :param hidden_dim1: The number of neurons in the first hidden layer
        :param hidden_dim2: The number of neurons in the second hidden layer
        :param latent_dim: The latent dimension (number of neurons in the bottleneck layer)
        :param learning_rate: The learning rate for the optimizer
        :param max_training_steps: The maximum number of training steps (used for the scheduler)
        """
        super(VariationalAutoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.max_training_steps = max_training_steps
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Encoder: input -> hidden1 -> hidden2 -> latent (mu and log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.mu_layer = nn.Linear(hidden_dim2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim2, latent_dim)
        self.beta = 1
        # Decoder: latent -> hidden2 -> hidden1 -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )
        self.validation_step_outputs = np.zeros((0, latent_dim))
        self.test_step_outputs = np.zeros((0, latent_dim))

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return decoded, mu, log_var

    def loss_function(self, reconstructed, x, mu, log_var):
        # Reconstruction loss (MSE) with mean reduction
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')

        # KL Divergence with mean reduction
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed, mu, log_var = self.forward(x)
        recon_loss, kl_loss = self.loss_function(reconstructed, x, mu, log_var)
        loss= recon_loss + self.beta*kl_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed, mu, log_var = self.forward(x)
        recon_loss, kl_loss = self.loss_function(reconstructed, x, mu, log_var)
        val_loss = recon_loss + kl_loss
        # Save latent representations for plotting
        encoded = self.reparameterize(mu, log_var)
        self.validation_step_outputs = np.concatenate((self.validation_step_outputs, encoded.cpu().detach().numpy()),
                                                      axis=0)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def on_validation_epoch_end(self):
        self.plot_latent_space(self.validation_step_outputs[:MAX_PLOT_POINTS], mode='val')
        self.validation_step_outputs = np.zeros((0, self.latent_dim))

    def plot_latent_space(self, encoded, mode='val'):
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(encoded)

        for label_type in ['binary', 'multiclass']:
            plt.figure(figsize=(5, 5))
            if label_type == 'binary':
                labels = self.trainer.datamodule.get_binary_labels(mode)[:encoded.shape[0]]
                class_names = BINARY_CLASS_NAMES
                cmap = mcolors.ListedColormap(['royalblue', 'red'])
            else:
                labels = self.trainer.datamodule.get_multiclass_labels(mode)[:encoded.shape[0]]
                class_names = MULTIClASS_CLASS_NAMES
                cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))

            found_labels = np.unique(labels)
            labels_names = list(filter(lambda x: x in found_labels, class_names))
            labels_to_idx = {label: i for i, label in enumerate(labels_names)}
            colors = [cmap(labels_to_idx[label]) for label in labels]

            bounds = np.arange(len(np.unique(labels)) + 1) - 0.5
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ticks=np.arange(len(np.unique(labels))))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5)
            cbar.ax.set_yticklabels(labels_names)

            explained_variance = reducer.explained_variance_ratio_.tolist()
            explained_variance = [round(e, 3) for e in explained_variance]
            title = f'PCA projection of latent_dim - {label_type} - {mode}'
            plt.title(title + '\n' + f'Explained variance: {explained_variance}')
            self.logger.experiment.add_figure(title, plt.gcf(), global_step=self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        x = batch
        reconstructed, mu, log_var = self.forward(x)
        recon_loss, kl_loss = self.loss_function(reconstructed, x, mu, log_var)
        test_loss = recon_loss + kl_loss
        encoded = self.reparameterize(mu, log_var)
        self.test_step_outputs = np.concatenate((self.test_step_outputs, encoded.cpu().detach().numpy()), axis=0)
        self.log('test_loss', test_loss)
        self.log('test_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def on_test_epoch_end(self):
        self.plot_latent_space(self.test_step_outputs[:MAX_PLOT_POINTS], mode='test')
        self.test_step_outputs = np.zeros((0, self.latent_dim))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_training_steps,
                                                                        eta_min=1e-6),
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    # Initialize the DataModule
    BATCH_SIZE = 256
    HIDDEN_DIM_1 = 256
    HIDDEM_DIM_2 = 64
    LATENT_DIM = 16
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 4
    N_EPOCHS = 50
    DROPOUT_RATE = 0.1

    VERSION = "debug2"

    data_module = AutoencoderDataModule(batch_size=BATCH_SIZE,
                                        val_ratio=VAL_RATIO,
                                        num_workers=NUM_WORKERS)
    data_module.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = next(iter(data_module.train_dataloader()))
    input_dim = sample_batch.shape[1]

    print(f"VAE configuration: Input dimension: {input_dim}, Batch size: {BATCH_SIZE}, Validation ratio: {VAL_RATIO}")

    # Initialize Variational Autoencoder model
    model = VariationalAutoencoder(input_dim=input_dim,
                                   hidden_dim1=HIDDEN_DIM_1,
                                   hidden_dim2=HIDDEM_DIM_2,
                                   latent_dim=LATENT_DIM,
                                   max_training_steps=len(data_module.train_dataloader()) * N_EPOCHS,
                                   learning_rate=LEARNING_RATE,
                                   dropout_rate=DROPOUT_RATE)

    # Define the TensorBoard logger
    logger = pl.loggers.TensorBoardLogger(LOGS_DIR, name="vae", version=VERSION)

    # Define the ModelCheckpoint callback (Save the best model based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOGS_DIR / 'vae' / VERSION,
        filename='vae-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        save_last=True,
        mode='min'
    )

    # Define the LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Define the Trainer, enabling TensorBoard logging and specifying the maximum epochs
    trainer = pl.Trainer(
        accelerator="auto",
        devices='auto',
        strategy="auto",
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=N_EPOCHS,
    num_sanity_val_steps=0
    )

    # Train the model
    trainer.fit(model, data_module, ckpt_path="last")

    # Test the model
    best_checkpoint = checkpoint_callback.best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint)