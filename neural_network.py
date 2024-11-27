import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Linear(256, input_dim),
            nn.Tanh()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def augment_with_vae(X_train, noise_factor=0.15, n_augmented=2500, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize VAE
    vae = VariationalAutoencoder(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    def loss_function(recon_x, x, mu, log_var):
        reconstruction_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + 0.1 * kld_loss
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for batch in loader:
            data = batch[0]
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Generate augmented samples
    vae.eval()
    augmented_samples = []
    
    with torch.no_grad():
        for _ in range(n_augmented):
            # Sample from latent space
            z = torch.randn(1, 16).to(device)
            # Add controlled noise
            z += noise_factor * torch.randn_like(z)
            # Generate new sample
            sample = vae.decoder(z).cpu().numpy()
            augmented_samples.append(sample[0])
    
    augmented_array = np.array(augmented_samples)
    return np.vstack([X_train, augmented_array])