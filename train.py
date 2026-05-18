import os
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BlipForConditionalGeneration
from tqdm.auto import tqdm

import config
from dataset import get_data_loaders

def main():
    print("="*50)
    print("AI Image Captioning - BLIP Fine-Tuning Pipeline")
    print("="*50)
    
    # ---------------------------------------------------------
    # 1. SETUP DEVICE & CONFIGURATION
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Configuration & Device")
    print(f"    Target Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
        
    debug_str = "ON (Fast Experimentation)" if getattr(config, 'DEBUG_MODE', False) else "OFF (Full Training)"
    print(f"    Debug Mode: {debug_str}")
    print(f"    Batch Size: {config.BATCH_SIZE}")
    print(f"    Gradient Accumulation: {getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)}")
    print(f"    Epochs: {config.EPOCHS}")
    print(f"    Learning Rate: {config.LEARNING_RATE}")
    
    # Ensure the model save directory exists
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # 2. PREPARE DATA
    # ---------------------------------------------------------
    print("\n[2] Preparing Datasets and DataLoaders...")
    train_loader, val_loader = get_data_loaders(config)
    
    if len(train_loader) == 0:
        print("\n[ERROR] Training DataLoader is empty! Please add data to your dataset folder.")
        return
        
    print(f"    Training Samples: {len(train_loader.dataset)}")
    print(f"    Validation Samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"    Training batches: {len(train_loader)}")
    print(f"    Validation batches: {len(val_loader)}")
    
    # ---------------------------------------------------------
    # 3. INITIALIZE MODEL & OPTIMIZER
    # ---------------------------------------------------------
    start_epoch = 1
    start_batch = 0
    resume_checkpoint = getattr(config, 'RESUME_CHECKPOINT', None)
    
    print("\n[3] Initializing BLIP Model...")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print("Resuming training from checkpoint...")
        model = BlipForConditionalGeneration.from_pretrained(resume_checkpoint)
    else:
        print("    -> Starting fresh training from base model")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Modern PyTorch AMP (Automatic Mixed Precision)
    scaler = GradScaler('cuda')
    grad_accum_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
    
    # Restore optimizer, scaler, and epoch state if resuming
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        state_path = os.path.join(resume_checkpoint, "training_state.pt")
        if os.path.exists(state_path):
            print("    -> Restoring optimizer and scaler states...")
            # map_location=device ensures tensors are loaded onto the correct GPU/CPU
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            
            start_epoch = state['epoch']
            if 'batch' in state:
                start_batch = state['batch'] + 1
                print(f"    -> Resuming training from Epoch {start_epoch}, Batch {start_batch}")
            else:
                start_epoch += 1
                start_batch = 0
                print(f"    -> Resuming training from Epoch {start_epoch}")
        else:
            print("    [WARNING] training_state.pt not found. Only model weights were restored.")
    
    # ---------------------------------------------------------
    # 4. TRAINING LOOP
    # ---------------------------------------------------------
    print("\n[4] Starting Training...")
    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        
        model.train()
        train_loss = 0.0
        
        # Zero gradients at the very beginning of the epoch
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue
                
            # Step 1: Move inputs to GPU
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Step 2: FORWARD PASS with Modern Mixed Precision
            with autocast('cuda'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Divide loss by accumulation steps so the gradients scale properly
                loss = outputs.loss / grad_accum_steps
            
            # Step 3: BACKWARD PASS
            scaler.scale(loss).backward()
            
            # Step 4: OPTIMIZER STEP (Gradient Accumulation)
            # Only update weights after N steps
            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate full loss for accurate logging
            train_loss += (loss.item() * grad_accum_steps)
            train_pbar.set_postfix({"batch_loss": f"{(loss.item() * grad_accum_steps):.4f}"})
            
            # Step 5: SAVE MID-EPOCH CHECKPOINT
            if hasattr(config, 'CHECKPOINT_SAVE_INTERVAL') and (batch_idx + 1) % config.CHECKPOINT_SAVE_INTERVAL == 0:
                mid_epoch_save_path = os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch}_batch_{batch_idx + 1}")
                os.makedirs(mid_epoch_save_path, exist_ok=True)
                print(f"\n    -> Saving mid-epoch checkpoint: {mid_epoch_save_path}")
                
                model.save_pretrained(mid_epoch_save_path)
                
                training_state = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }
                torch.save(training_state, os.path.join(mid_epoch_save_path, "training_state.pt"))
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"    -> Average Training Loss: {avg_train_loss:.4f}")
        
        # ---------------------------------------------------------
        # 5. VALIDATION LOOP
        # ---------------------------------------------------------
        if val_loader and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}")
                for batch in val_pbar:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    with autocast('cuda'):
                        outputs = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    val_loss += loss.item()
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
                    
            avg_val_loss = val_loss / len(val_loader)
            print(f"    -> Average Validation Loss: {avg_val_loss:.4f}")
        
        # ---------------------------------------------------------
        # 6. SAVE MODEL CHECKPOINT
        # ---------------------------------------------------------
        epoch_save_path = os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch}")
        os.makedirs(epoch_save_path, exist_ok=True)
        print(f"    -> Saving checkpoint to: {epoch_save_path}")
        
        # HuggingFace standard save method (saves pytorch_model.bin and config.json)
        model.save_pretrained(epoch_save_path)
        
        # Save PyTorch states for easy resuming
        training_state = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(training_state, os.path.join(epoch_save_path, "training_state.pt"))
        
    print("\n[SUCCESS] Fine-tuning completed!")

if __name__ == "__main__":
    main()


