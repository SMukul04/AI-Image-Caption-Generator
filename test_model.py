from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

from PIL import Image
from gtts import gTTS

import torch
import os

# -------------------------
# GPU Setup
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {device}")

# -------------------------
# Load BLIP Model
# -------------------------

print("Loading model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Model loaded successfully!")

# -------------------------
# Load Image
# -------------------------

image = Image.open("images/boy2.jpg").convert("RGB")

# -------------------------
# Process Image
# -------------------------

inputs = processor(
    image,
    return_tensors="pt"
).to(device)

# -------------------------
# Generate Caption
# -------------------------

print("Generating caption...")

output = model.generate(**inputs)

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

# -------------------------
# Print Caption
# -------------------------

print("\nGenerated Caption:")
print(caption)

# -------------------------
# Convert Caption To Speech
# -------------------------

tts = gTTS(text=caption, lang='en')

tts.save("caption.mp3")

print("\nPlaying Audio Caption...")

os.system("start caption.mp3")