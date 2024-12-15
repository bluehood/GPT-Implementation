import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from tokeniser import load_tokeniser, encode_text, decode_text, clean_dataset
from model import GPT, GPTConfig 

from utils import print_model_structure

class TextDataset(Dataset):
    def __init__(self, text_path, tokeniser, seq_length=128):
        self.tokeniser = tokeniser
        self.seq_length = seq_length
        
        with open(text_path, "r") as f:
            text = f.read()

        text = clean_dataset(text)
        encoded = encode_text(text, self.tokeniser)
        
        self.data = torch.LongTensor(encoded)
        self.n_sequences = len(self.data) - seq_length

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        
        return sequence, target

def create_dataloader(text_path, tokeniser, batch_size=64, seq_length=128, num_workers=4):
    tokeniser = load_tokeniser(tokeniser)
    dataset = TextDataset(text_path, tokeniser, seq_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, tokeniser

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (sequences, targets) in enumerate(progress_bar):
        # Move data to device
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, loss = model(sequences, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, f'{save_dir}/checkpoint_epoch_{epoch}.pt')

def main():
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    batch_size = 64
    seq_length = 128
    learning_rate = 3e-4
    save_dir = './models/checkpoints'
    
    # Load tokenizer and create dataloader
    tokeniser_path = "./models/alice_in_wonderland_tokeniser.json"
    dataloader, tokeniser = create_dataloader(
        "./datasets/alice_in_wonderland.txt",
        tokeniser_path,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Initialize model
    config = GPTConfig(
        vocab_size=len(tokeniser),
        block_size=seq_length,
        n_layer=6,
        n_embd=384,             
        num_heads=6,
        dropout=0.1
    )
    
    model = GPT(config)
    model = model.to(device)
    
    print('-' * 50)
    print_model_structure(model, config)
    print('-' * 50)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(dataloader),
        eta_min=1e-5
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, epoch)
        
        # Save checkpoint if best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, save_dir)
            

def debug():
    import debugpy
    debugpy.listen(('127.0.0.1', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    return

if __name__ == "__main__":
    debug()
    main()