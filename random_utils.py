import torch
import wandb
import os
import gc
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import save_model


# Model parameter count
def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


# basic training loop
def training_loop(
    model, train_loader, epochs, config, optimizer, criterion=nn.CrossEntropyLoss
):
    scaler = GradScaler()
    model.train()
    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = model(image)
                train_loss = criterion(output, label.long())
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            wandb.log({"loss": train_loss})

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        print(f"Epoch @ {epoch} complete!")

    print(f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    safe_tensorfile = save_model(model, config.safetensor_file)

    torch.save(model.state_dict(), f"{config.model_file}")


# training_loop()

torch.cuda.empty_cache()
gc.collect()

# Ciao
