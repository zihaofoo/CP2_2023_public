import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from utils_public import *
import seaborn as sns
import pandas as pd


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

def score_samples(samples): #Function to score all samples. Requires trained regressors in all_predictors object
    samples = samples.reshape(samples.shape[0], 49) #Reformat data into the format regressors expect
    samples = pd.DataFrame(samples, columns = range(grids_subset.shape[1]), dtype = "object")
    sample_predictions = []
    for i in range(4): #Loop over advisores
        predictor = all_predictors[i] #Select appropriate regressor
        sample_predictions.append(predictor.predict(samples)) #Call regressor
    sample_predictions = np.stack(sample_predictions).T #Stack scores together
    return sample_predictions

def compare_violinplots(all_predictions, all_names):
    #Wrangle dataframe predictions into the format expected by seaborn violinplot
    all_dfs = []
    for i in range(len(all_predictions)):
        df = pd.DataFrame(all_predictions[i], columns=["Wellness", "Tax", "Transportation", "Business"])
        df["Minimum Score"] = np.min(df.values, axis=1) #Calculate minimum score over the four advisors for every grid
        df = pd.melt(df, var_name = "Advisor", value_name = "Score")
        df['Method'] = all_names[i]
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs, axis=0)
    plt.figure(figsize=(10,5))

    #Plot the distributions
    sns.violinplot(x="Advisor", y="Score", hue="Method", data=all_dfs, linewidth=1, palette = ["#F08E18", "#888888", "#DC267F"])

grids = load_grids() #Helper function we have provided to load the grids from the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu is available, otherwise use cpu
grids_oh = (np.arange(5) == grids[...,None]).astype(int) # Onehot encode
grids_tensor = torch.from_numpy(grids_oh) # Torch tensor from numpy
grids_tensor = grids_tensor.permute(0, 3, 1, 2) # Reshape to organize data by [batch, district, x, y]
grids_tensor = grids_tensor.float() # Ensure we are using floats
grids_tensor = grids_tensor.to(device) # Send the tensor to device

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

model = VAE(input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride).to(device) #Instantiate the VAE
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Instantiate the Optimizer

summary(model, input_size=(input_channels, image_size[0], image_size[1]))

# Main loop
for epoch in range(1, num_epochs + 1): #Loop over num_epochs
    train(epoch, grids_tensor) #Call train function for each epoch


originals = np.random.choice(np.arange(len(grids)), size=5, replace=False) #Select 5 random indices
reconstructions = reconstruct_from_vae(model, grids_tensor[originals], device) #Reconstruct
plot_reconstruction(grids[originals], reconstructions) #Compare

samples = sample_from_vae(model, 7, latent_dim, device)
plot_n_grids(samples) #Plot generated grids

generated_samples = sample_from_vae(model, 1000, latent_dim, device) #Sample from VAE
random_samples = np.random.choice(np.arange(5), size = (1000,7,7)) #Randomly Sample Grids

generated_sample_predictions = score_samples(generated_samples)
random_sample_predictions = score_samples(random_samples)


all_predictions = [generated_sample_predictions, random_sample_predictions, final_prediction_array[:1000]] #Select only the last 1000 of the dataset for speed
all_names = ["VAE-Generated", "Random-Generated", "Dataset Subset"]
compare_violinplots(all_predictions, all_names) #Plot!