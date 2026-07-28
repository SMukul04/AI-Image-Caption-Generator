import os

# ==========================================
# Paths and Directories
# ==========================================
# The base directory where the dataset is stored
DATASET_BASE_DIR = os.path.join("dataset", "flickr8k")

# Path to the directory containing all raw images
IMAGES_DIR = os.path.join(DATASET_BASE_DIR, "images")

# Path to the text file containing image captions
CAPTIONS_FILE = os.path.join(DATASET_BASE_DIR, "captions", "captions.txt")

# Paths to the text files containing the splits (train, validation, test)
SPLITS_DIR = os.path.join(DATASET_BASE_DIR, "splits")
TRAIN_SPLIT_FILE = os.path.join(SPLITS_DIR, "train.txt")
VAL_SPLIT_FILE = os.path.join(SPLITS_DIR, "val.txt")
TEST_SPLIT_FILE = os.path.join(SPLITS_DIR, "test.txt")

# ==========================================
# Model & Training Configuration
# ==========================================
# These are placeholder hyperparameters for when we actually start training.
# Feel free to tweak these later depending on your hardware limits.

# The number of images to process at the same time
BATCH_SIZE = 2

# The size to which every image will be resized before passing to the model
IMAGE_SIZE = (224, 224)

# Number of complete passes through the dataset
EPOCHS = 1

# How much the model should update its weights during training (learning rate)
# Using 1e-5 to prevent catastrophic forgetting (1e-4 is too high for BLIP)
LEARNING_RATE = 1e-5

# Frequency of printing training progress (e.g. print every 10 batches)
LOG_INTERVAL = 10

# Frequency of saving mid-epoch checkpoints (in batches)
CHECKPOINT_SAVE_INTERVAL = 100

# ==========================================
# Debugging & Optimization
# ==========================================
# Set to True for fast experimentation
DEBUG_MODE = False
DEBUG_TRAIN_SAMPLES = 100
DEBUG_VAL_SAMPLES = 20

# Number of CPU cores to use for loading images (0 = main thread only, 4 is usually good)
NUM_WORKERS = 4

# Simulate larger batch sizes (e.g. batch_size 4 * grad_accum 4 = effective batch size 16)
GRADIENT_ACCUMULATION_STEPS = 2

# Directory to save model checkpoints after each epoch
MODEL_SAVE_DIR = os.path.join("model", "checkpoints")

RESUME_CHECKPOINT = "model/checkpoints/epoch_1_batch_2900"

# The path to your fine-tuned model for hybrid caption generation in the Flask app
FINETUNED_MODEL_PATH = "model/final_model"