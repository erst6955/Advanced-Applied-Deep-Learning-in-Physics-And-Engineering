

import torchvision
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import random



# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
batchSize = 32
logStep = 625
latent_dimension = 128
image_dimension = 784

myTransforms = transforms.Compose([ # we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
    transforms.ToTensor(),          # which will convert the image range from [0, 1] to [-1, 1]
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)


class Generator(nn.Module):
    """
    Generator Model which takes a random noise vector and generates a fake image. The input noise in latent space is an abstract space where each point corresponds to a different kind of image.
    The noise vector doesn't directly mean anything in the beginning, but after training, it gets mapped by the generator to a specific kind of image — like a "7", or a "2", or a "3". We make
    assumptions about how the input noise is distributed.
    """
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential( # simple NN
            nn.Linear(latent_dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, image_dimension),
            nn.Tanh(), # tanh activation function to get the output in the range of [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    """
    Discriminator Model which takes an image and outputs a probability of it being real or fake. Furthermore,
    the discriminator is a binary classifier which uses the sigmoid activation and the BCE loss function.
    """
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # add this
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3), # add this
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        return self.disc(x)
    

def train_models(discriminator, generator, fixed_noise, num_examples, opt_discriminator, opt_generator, criterion, epochs, writer):
    """
    Trains the GAN models using the MNIST dataset.
    The generator uses BCE where 1 is the label for fake images and 0 is the label for real images. This mean that it learns
    to generate fake images that are similar to the real images in the dataset since the disciminator has real images labeled as
    1 and fake images labeled as 0. I.e., the generator aims to minimize the loss by producing images that are classified as real by the discriminator.
    The disciminator on the other hand is simply trained to identify the real and fake images. 
    """
    step = 0 # for tensorboard logging
    gen_losses = []
    disc_losses = []

    # ====================== Training Loop ========================================
    for epoch in range(epochs):
    
        start_time = time.time()  
    
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, image_dimension).to(device) # flatten the image in order to pass it to the discriminator
            batch_size = real.shape[0] # get the batch size

            noise = torch.randn(batch_size, latent_dimension).to(device) # generate random noise from a normal distribution
            fake = generator(noise) # generate fake images from the noise

            # Discriminator Loss
            disc_real = discriminator(real).view(-1) # get the discriminator output for real images
            loss_real = criterion(disc_real, torch.full_like(disc_real, 0.9)) # pass through BCE where real images are labeled as 1

            disc_fake = discriminator(fake.detach()).view(-1) # get the discriminator output for fake images
            loss_fake = criterion(disc_fake, torch.full_like(disc_fake, 0.1)) # pass through BCE where fake images are labeled as 0

            loss_discriminator = (loss_real + loss_fake) / 2 # average the loss for real and fake images

            discriminator.zero_grad() # zero the gradients
            loss_discriminator.backward(retain_graph=True) # backpropagate the loss
            opt_discriminator.step() # update the discriminator weights

            # ===== Generator Loss =====
            # The generator tries to fool the discriminator, so we want the discriminator to think that the fake images are real
            output = discriminator(fake).view(-1) # get the discriminator output for fake images
            loss_generator = criterion(output, torch.ones_like(output)) # pass through BCE where fake images are labeled as 1
            # The generator tries to maximize the probability of the discriminator being wrong

            generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            if batch_idx % logStep == 0: # tensorboard logging
                with torch.no_grad():
                    fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28) # reshape the fake images to 1 channel and 28x28
                    real_images = real.reshape(-1, 1, 28, 28) # reshape the real images to 1 channel and 28x28
 
                    imgGridFake = torchvision.utils.make_grid(fake_images, normalize=True) # make a grid of fake images
                    imgGridReal = torchvision.utils.make_grid(real_images, normalize=True) # make a grid of real images

                    # Denormalize: [-1, 1] → [0, 1]
                    fake_images = denormalize(fake_images) 
                    real_images = denormalize(fake_images) 

                    writer.add_image("MNIST Fake Images", imgGridFake, global_step=step)
                    writer.add_image("MNIST Real Images", imgGridReal, global_step=step)
                    writer.add_scalar("Loss Discriminator", loss_discriminator.item(), step)
                    writer.add_scalar("Loss Generator", loss_generator.item(), step)

                    step += 1
        for param_group in opt_generator.param_groups:
            lr_g = param_group['lr']
        for param_group in opt_discriminator.param_groups:
            lr_d = param_group['lr']
        
        # At the end of each epoch
        gen_losses.append(loss_generator.item())
        disc_losses.append(loss_discriminator.item())
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_discriminator:.4f} | lr D: {lr_d} | Loss G: {loss_generator:.4f} | lr G: {lr_g} | {elapsed:.2f}s")

        get_fake_images(generator, fixed_noise, latent_dimension, num_examples, epoch) # get some fake images for the epoch

    return gen_losses, disc_losses
        

def denormalize(images):
    return images * 0.5 + 0.5



def get_fake_images(generator, fixed_noise, latent_dimension, num_examples, epoch):
    """
    Returns a batch of fake images from the generator.
    """

    generator.eval()

    # Generate images from noise
    with torch.no_grad():
        generated_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
        generated_images = denormalize(generated_images) # denormalize the images

    # Create a grid for visualization
    grid = torchvision.utils.make_grid(generated_images.cpu(), nrow=4, normalize=True)

    # Plot the generated images
    plt.figure(figsize=(8, 8))
    plt.title(f"Fake Images Generated by GAN at epoch {epoch+1}")
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"Figures/fake_images_{epoch+1}.png", bbox_inches='tight')
    plt.close()



def plot_losses(gen_losses, disc_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/loss_plot.png", dpi=300)



# =========================================== Run Code =========================================================
discriminator = Discriminator().to(device)
generator = Generator().to(device)

lr_G = 1e-4
lr_D = 5e-5

opt_generator = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
criterion = nn.BCELoss() # Binary Cross Entropy Loss

epochs = 30
num_examples = 3

fixed_noise = torch.randn(num_examples, latent_dimension).to(device) # fixed noise for reproducibility after every epoch
writer = SummaryWriter("runs/GAN_MNIST")

gen_losses, disc_losses = train_models(discriminator, generator, fixed_noise, num_examples, opt_discriminator, opt_generator, criterion, epochs, writer)
plot_losses(gen_losses, disc_losses) 

writer.close()
print("Complete!")







