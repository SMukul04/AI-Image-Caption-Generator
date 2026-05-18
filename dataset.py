import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor

# In PyTorch, we create a custom Dataset class to handle loading our specific data.
# This class needs to tell PyTorch how to get a single item (image + caption) from the dataset.

class Flickr8kDataset(Dataset):
    """
    Custom Dataset class for Flickr8k.
    This class handles reading images from disk, applying BLIP transformations 
    (like resizing and normalization), and tokenizing the corresponding captions.
    """
    def __init__(self, images_dir, captions_file, processor, split_file=None, max_samples=None, use_all_captions=False):
        """
        Initialization:
        - images_dir: Path to the folder with all images.
        - captions_file: Path to the file containing image captions.
        - processor: The BLIP processor for text and image preparation.
        - split_file: Optional path to a file containing a list of image names for a specific split (train/val/test).
        """
        self.images_dir = images_dir
        self.processor = processor
        
        # Load all captions into a dictionary: { "image_name.jpg": ["caption 1", "caption 2", ...] }
        self.captions_dict = self._load_captions(captions_file)
        
        # If a split file is provided, filter the dataset to only include those images.
        # Otherwise, use all images found in the captions dictionary.
        if split_file and os.path.exists(split_file):
            self.image_names = self._load_split(split_file)
        else:
            self.image_names = list(self.captions_dict.keys())
            
        # Create a flattened list of (image_name, single_caption) pairs.
        # Since Flickr8k has multiple captions per image, we pair each image with every caption it has.
        self.dataset_pairs = []
        for img_name in self.image_names:
            if img_name in self.captions_dict:
                if use_all_captions:
                    for caption in self.captions_dict[img_name]:
                        self.dataset_pairs.append((img_name, caption))
                else:
                    # Deterministically use only the first caption
                    self.dataset_pairs.append((img_name, self.captions_dict[img_name][0]))
                    
        # Truncate dataset if max_samples is provided (useful for debug mode)
        if max_samples is not None:
            self.dataset_pairs = self.dataset_pairs[:max_samples]
                    
    def _load_captions(self, captions_file):
        """Helper function to read the captions text file."""
        captions_dict = {}
        if not os.path.exists(captions_file):
            print(f"Warning: Captions file not found at {captions_file}. Please add the dataset.")
            return captions_dict
            
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Typically Flickr8k captions are in format: image_name.jpg,This is a caption
                # We skip the header if there is one.
                if line.strip() == "" or "image,caption" in line.lower():
                    continue
                
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    caption = ",".join(parts[1:]) # Rejoin in case caption has commas
                    
                    if img_name not in captions_dict:
                        captions_dict[img_name] = []
                    captions_dict[img_name].append(caption)
        return captions_dict

    def _load_split(self, split_file):
        """Helper function to read the train/val/test text files."""
        with open(split_file, 'r') as f:
            # Read all lines and strip whitespace/newlines
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        """Returns the total number of (image, caption) pairs in this dataset."""
        return len(self.dataset_pairs)

    def __getitem__(self, idx):
        """
        Fetches one pair of data at the given index.
        This is called by the DataLoader during training.
        """
        img_name, caption = self.dataset_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load the image using PIL (Python Imaging Library)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # If image is missing, we create a dummy black image
            image = Image.new('RGB', (224, 224), color='black')
        
        # ---------------------------------------------------------
        # PREPROCESSING WITH BLIP PROCESSOR
        # The processor handles resizing, normalizing, and converting
        # the image to a tensor, as well as tokenizing the caption.
        # padding="max_length" ensures all captions have the same length.
        # truncation=True ensures captions don't exceed max length.
        # ---------------------------------------------------------
        encoding = self.processor(
            images=image, 
            text=caption, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the processor
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # The 'input_ids' from the text act as our labels for training.
        # We clone it so we can modify it if needed during training.
        item["labels"] = item["input_ids"].clone()
        
        # Return a dictionary containing processed image tensors, tokenized captions, and labels
        return item

def get_data_loaders(config):
    """
    A helper function to create DataLoaders for train and validation.
    This encapsulates the setup logic so main training script stays clean.
    """
    
    print("    Initializing BLIP Processor...")
    # Load the processor for Salesforce/blip-image-captioning-base
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Determine sample limits for Debug Mode
    train_max = config.DEBUG_TRAIN_SAMPLES if getattr(config, 'DEBUG_MODE', False) else None
    val_max = config.DEBUG_VAL_SAMPLES if getattr(config, 'DEBUG_MODE', False) else None
    
    use_all_captions = getattr(config, 'USE_ALL_CAPTIONS', False)
    
    # 1. Create the Dataset objects with the initialized processor
    train_dataset = Flickr8kDataset(
        images_dir=config.IMAGES_DIR, 
        captions_file=config.CAPTIONS_FILE, 
        processor=processor,
        split_file=config.TRAIN_SPLIT_FILE,
        max_samples=train_max,
        use_all_captions=use_all_captions
    )
    
    val_dataset = Flickr8kDataset(
        images_dir=config.IMAGES_DIR, 
        captions_file=config.CAPTIONS_FILE, 
        processor=processor,
        split_file=config.VAL_SPLIT_FILE,
        max_samples=val_max,
        use_all_captions=use_all_captions
    )
    
    # GPU-friendly loading: pin_memory=True speeds up data transfer from CPU to GPU.
    pin_memory = torch.cuda.is_available()
    num_workers = getattr(config, 'NUM_WORKERS', 0)
    persistent_workers = num_workers > 0
    
    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers
        )
    except ValueError:
        # Happens if the dataset is completely empty
        train_loader = []
    
    try:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers
        )
    except ValueError:
        val_loader = []
    
    return train_loader, val_loader
