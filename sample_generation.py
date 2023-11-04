import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from utils_public import *
import seaborn as sns
import pandas as pd
from CNN_NIC_sub_v2 import *
from tensorflow.keras.models import load_model
from matplotlib.pyplot import hist


class VAE(nn.Module): #Create VAE class inheriting from pytorch nn Module class
    def __init__(self, input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride):
        super(VAE, self).__init__()

        # Create encoder model
        self.encoder = Encoder(input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride)

        #Create decoder after calculating input size for decoder
        decoder_input_size = self.calculate_decoder_input_size(image_size, num_layers, kernel_size, stride)
        self.decoder = Decoder(input_channels, hidden_size, num_layers, latent_dim, decoder_input_size, kernel_size, stride)

    def calculate_decoder_input_size(self, image_size, num_layers, kernel_size, stride):
        #Function to calculate the input size of the decoder given its architecture
        h, w = image_size
        for _ in range(num_layers):
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
        return h, w

    def reparameterize(self, mu, logvar):
        #Sample from gaussian
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        #Pass through encoder, reparameterize using mu and logvar as given by the encoder, then pass through decoder
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

class Encoder(nn.Module): #Encoder model of VAE
    def __init__(self, input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride):
        super(Encoder, self).__init__()

        layers = []
        h, w = image_size
        in_channels = input_channels
        for _ in range(num_layers): # Loop over layers, adding conv2d, layernorm, and relu.
            h = (h - kernel_size) // stride + 1 #Update h and w to compensate for previous layers output
            w = (w - kernel_size) // stride + 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_size, kernel_size, stride),
                    nn.LayerNorm([hidden_size, h, w]),
                    nn.ReLU()
                )
            )
            in_channels = hidden_size #Input channels to later conv layers will just be the hidden size

        self.conv_layers = nn.ModuleList(layers) #Collect convolution layers and layernorm in conv_layers object
        self.final_flatten_size = h * w * hidden_size #Calculate size of final FC output layer
        self.fc_mu = nn.Linear(self.final_flatten_size, latent_dim) #Final FC layer to output mean
        self.fc_logvar = nn.Linear(self.final_flatten_size, latent_dim) #Final FC layer to output logvar

    def forward(self, x): #Forward call for encoder
        for layer in self.conv_layers: #Call conv layers sequentially
            x = layer(x)
        x = x.view(x.size(0), -1) #Flatten x
        mu = self.fc_mu(x) #Get mu and logvar from FC layers
        logvar = self.fc_logvar(x)
        return mu, logvar #Return mu and logvar

class Decoder(nn.Module):  #Decoder model of VAE
    def __init__(self, output_channels, hidden_size, num_layers, latent_dim, decoder_input_size, kernel_size, stride):
        super(Decoder, self).__init__()
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size

        #Initial fully connected layer
        self.fc = nn.Linear(latent_dim, hidden_size * decoder_input_size[0] * decoder_input_size[1])
        layers = []
        h, w = decoder_input_size
        for _ in range(num_layers-1): # Loop over layers, adding conv2dtranspose, layernorm, and relu.
            h = (h - 1) * stride + kernel_size #Update h and w to compensate for previous layers output
            w = (w - 1) * stride + kernel_size
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size, stride),
                    nn.LayerNorm([hidden_size, h, w]),
                    nn.ReLU()
                )
            )

        self.deconv_layers = nn.ModuleList(layers) #Collect deconv layers

        #Final layer brings the image to the original size
        self.final_layer = nn.ConvTranspose2d(hidden_size, output_channels, kernel_size, stride)

    def forward(self, z):
        z = self.fc(z) #Call initial FC layer
        z = z.view(z.size(0), self.hidden_size, self.decoder_input_size[0], self.decoder_input_size[1])  # Reshape to match the deconvolution input shape
        for layer in self.deconv_layers: #Sequentially call deconv layers
            z = layer(z)
        z = self.final_layer(z)
        return torch.sigmoid(z) #Final sigmoid layer

def loss_function(recon_x, x, mu, logvar):
    # VAE loss is a sum of KL Divergence regularizing the latent space and reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') # Reconstruction loss from Binary Cross Entropy
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL Divergence loss
    return BCE + KLD

def train(epoch, data_tensor): #Train function for one epoch of training
    model.train()
    train_loss = 0
    num_batches = len(data_tensor) // batch_size

    #Tqdm progress bar object contains a list of the batch indices to train over
    progress_bar = tqdm(range(num_batches), desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)

    for batch_idx in progress_bar: #Loop over batch indices
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        data = data_tensor[start_idx:end_idx] #Gather corresponding data

        optimizer.zero_grad() #Set up optimizer
        recon_batch, mu, logvar = model(data) #Call model
        loss = loss_function(recon_batch, data, mu, logvar) #Call loss function
        loss.backward() #Get gradients of loss
        train_loss += loss.item() #Append to total loss
        optimizer.step() #Update weights using optimizeer

        # Updating the progress bar
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

    average_train_loss = train_loss / len(data_tensor) #Calculate average train loss
    tqdm.write('Epoch: {} \tTraining Loss: {:.3f}'.format(epoch, average_train_loss))

def argmax_sample(grids): #Expects a BATCH of one-hot encoded grids (nx7x7x5)
    return torch.argmax(grids, axis=3)

def probabilistic_sample(grids): #Expects a BATCH of one-hot encoded grids (nx7x7x5)
    flattened_grids = grids.reshape(grids.shape[0]*7*7, 5)
    flattened_grids = torch.multinomial(flattened_grids, num_samples=1)
    grids = flattened_grids.reshape(grids.shape[0],7,7)
    return grids

def reconstruct_from_vae(model, samples, device='cpu'):
    #Function to reconstruct city grids
    with torch.no_grad(): #Faster inference if model does not need to calculate gradients
        samples = model(samples)[0] #Pass samples through VAE
        samples = samples.permute(0, 2, 3, 1) #Reshuffle dimensions to be [batch, x, y, district]
        samples = argmax_sample(samples) #Can switch to probabilistic sample
    return samples.to('cpu').numpy()

def plot_reconstruction(originals, reconstructions):
    # Function to plot reconstructed city grids alongside originals
    n = len(originals)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(9, 4*n))
    for i in range(n): # Loop over the grids
        plot_grid_image(originals[i], on_ax=axes[i, 0]) # Plot original on the left
        plot_grid_image(reconstructions[i], on_ax=axes[i, 1]) #Plot reconstructed on the right
    fig.tight_layout()
    plt.show()

def sample_from_vae(model, num_samples, latent_dim, device='cpu'):
    #Function to generate new samples from VAE
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device) #Sample from N(0,1) Gaussian
        samples = model.decoder(z) #Pass noise into decoder to sample a batch of  grid
        samples = samples.permute(0, 2, 3, 1) #Reshuffle dimensions to be [batch, x, y, district]
        samples = probabilistic_sample(samples) #Can switch to argmax sample
    return samples.to('cpu').numpy()

def score_samples(grids): #Function to score all samples. Requires trained regressors in all_predictors object
    grids_onehot = np.array([one_hot_encode(grid) for grid in grids])
    grids_onehot = grids_onehot.astype(np.float64)

    features0 = []
    features1 = []
    features2 = []
    features3 = []

    for grid in grids:
        features = compute_features(grid, advisor = 1)
        features = np.nan_to_num(features, nan = 0)
        features1.append(features)
    features1 = np.array(features1)
    features1[np.isnan(features1)] = 0
    features1 = features1.astype(np.float64)
   
    model1 = load_model("model1.h5")
    preds1 = model1.predict([grids_onehot, features1])
    if preds1 <0.85:
        return 0,0,0,0
    
    for grid in grids:
        features = compute_features(grid, advisor = 3)
        features = np.nan_to_num(features, nan = 0)
        features3.append(features)
    features3 = np.array(features3)
    features3[np.isnan(features3)] = 0   
    features3 = features3.astype(np.float64)
    
    model3 = load_model("model3.h5")
    preds3 = model3.predict([grids_onehot, features3])
    if preds3 <0.85:
        return 0,0,0,0
    
    for grid in grids:
        features = compute_features(grid, advisor = 0)
        features = np.nan_to_num(features, nan = 0)
        features0.append(features)
    features0 = np.array(features0)
    features0[np.isnan(features0)] = 0
    features0 = features0.astype(np.float64)
    model0 = load_model("model0.h5")
    preds0 = model0.predict([grids_onehot, features0])
    if preds0 <0.85:
        return 0,0,0,0
    
    for grid in grids:
        features = compute_features(grid, advisor = 2)
        features = np.nan_to_num(features, nan = 0)
        features2.append(features)
    features2 = np.array(features2)
    features2[np.isnan(features2)] = 0
    features2 = features2.astype(np.float64)

    model2 = load_model("model2.h5")
    preds2 = model2.predict([grids_onehot, features2])


    return preds0, preds1, preds2, preds3

#Keep fixed for 7x7 grid with 5 district options
input_channels = 5
image_size = (7, 7)

#Can tune these parameters
latent_dim = 20
hidden_size = 128
num_layers = 2
kernel_size = 3
stride = 1
num_epochs = 60
batch_size = 1024

VAE_model = torch.load('VAE_model' + '.pth')
VAE_model.eval()  # Don't forget to call eval() for inference

num_sample = 1
num_accept = 0

while num_accept < 100:
    
    generated_samples = sample_from_vae(VAE_model, num_sample, latent_dim) #Sample from VAE
    random_samples = np.random.choice(np.arange(5), size = (num_sample,7,7)) #Randomly Sample Grids

    preds0, preds1, preds2, preds3 = score_samples(generated_samples)
    rand_preds0, rand_preds1, rand_preds2, rand_preds3 = score_samples(random_samples)

    if preds0 > 0.85 and preds1 > 0.85 and preds2 > 0.80 and preds3 > 0.85:
        if num_accept == 1:
            np.save('samples_VAE.npy', generated_samples)     
        else:
            loaded_data = np.load('samples_VAE.npy')
            new_data = np.vstack(loaded_data, generated_samples)
            np.save('samples_VAE.npy', new_data)
        num_accept += 1

    print(num_accept, ":", preds0, preds1, preds2, preds3)
    
    if rand_preds0 > 0.85 and rand_preds1 > 0.85 and rand_preds2 > 0.80 and rand_preds3 > 0.85:
        if num_accept == 1:
            np.save('samples_rand.npy', random_samples)     
        else:
            loaded_data = np.load('samples_rand.npy')
            new_data = np.vstack(loaded_data, random_samples)
            np.save('samples_rand.npy', new_data)
        num_accept += 1

    print(num_accept, ":", rand_preds0, rand_preds1, rand_preds2, rand_preds3)


