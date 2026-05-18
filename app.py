import os
import time
import torch
import difflib
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import config

app = Flask(__name__)

# Configure the upload and audio folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
AUDIO_FOLDER = os.path.join('static', 'audio')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# CAPTION FUSION SYSTEM
# ---------------------------------------------------------
def fuse_captions(cap1, cap2):
    """
    Intelligently merges two captions by identifying the more detailed one
    and appending non-redundant meaningful phrases from the other.
    """
    import re
    if not cap1: return cap2.capitalize() if cap2 else ""
    if not cap2: return cap1.capitalize() if cap1 else ""
    
    c1 = cap1.lower().strip('.,!?"\' ')
    c2 = cap2.lower().strip('.,!?"\' ')
    if c1 == c2: return cap1.capitalize()
    
    w1 = c1.split()
    w2 = c2.split()
    
    # Safeguard against model collapse (gibberish repetition)
    def is_gibberish(words):
        if len(words) < 3: return False
        return len(set(words)) / len(words) < 0.4
        
    if is_gibberish(w2): return cap1.capitalize()
    if is_gibberish(w1): return cap2.capitalize()
    
    # Calculate Jaccard similarity
    set1, set2 = set(w1), set(w2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    sim = len(intersection) / len(union) if union else 0
    
    # If heavily overlapping, just pick the more detailed one
    if sim > 0.65:
        return (cap2 if len(w2) >= len(w1) else cap1).capitalize()
        
    primary = w2
    secondary = w1
    
    matcher = difflib.SequenceMatcher(None, primary, secondary)
    additions = []
    
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'with', 'in', 'on', 'at', 'to', 'of', 'for', 'her', 'his', 'their', 'my', 'your'}
    
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode in ('insert', 'replace'):
            chunk = secondary[b0:b1]
            meaningful = [w for w in chunk if w not in stop_words]
            
            # If the chunk is just replacing the main subject (usually early in sentence), skip it to avoid "woman" vs "girl" conflict
            if opcode == 'replace' and a0 < 5 and b0 < 5 and len(meaningful) <= 2:
                continue
                
            if meaningful:
                has_ing = any(w.endswith('ing') for w in chunk)
                # Keep if it has an action, or if it's a substantial detail
                if has_ing or len(meaningful) >= 2:
                    # Clean up leading conjunctions in chunk
                    if chunk[0] in ('and', 'with', 'while'):
                        chunk = chunk[1:]
                    if not chunk: continue
                    additions.append(" ".join(chunk))

    fused = " ".join(primary)
    
    if additions:
        for add in additions:
            words_in_add = add.split()
            if any(w.endswith('ing') for w in words_in_add):
                fused += " while " + add
            else:
                fused += " and " + add
                
    # Grammar Cleanup
    # Remove duplicate adjacent words
    fused = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', fused)
    
    # Fix common messy conjunctions
    replacements = [
        ("and and", "and"),
        ("while while", "while"),
        ("while and", "while"),
        ("and while", "while"),
        ("with and", "with"),
        ("with while", "while"),
        ("in in", "in"),
        ("on on", "on"),
        ("a a", "a"),
        ("the the", "the")
    ]
    for old, new in replacements:
        fused = fused.replace(old, new)
        
    return fused.capitalize()

# ---------------------------------------------------------
# LOAD DUAL BLIP MODELS
# ---------------------------------------------------------
print("Initializing Base and Fine-Tuned BLIP models globally. Please wait...")

# Use GPU if CUDA is available, otherwise default to internal CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Processing Device: {device}")

# 1. Load the universal processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Load the ORIGINAL Base Model
base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
print("✅ Original Base Model loaded.")

# 3. Load the FINE-TUNED Checkpoint Model (if it exists)
finetuned_model = None
if hasattr(config, 'FINETUNED_MODEL_PATH') and config.FINETUNED_MODEL_PATH and os.path.exists(config.FINETUNED_MODEL_PATH):
    finetuned_model = BlipForConditionalGeneration.from_pretrained(config.FINETUNED_MODEL_PATH).to(device)
    print(f"✅ Fine-Tuned Model loaded from {config.FINETUNED_MODEL_PATH}.")
else:
    print(f"⚠️ Fine-Tuned Model NOT FOUND at configured path. Using Base Model only.")

print("Models are ready to generate captions!")

def generate_hybrid_caption(image_path):
    """Processes an image through both models and returns the fused caption."""
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    
    # Generate kwargs to prevent stuttering
    gen_kwargs = {
        "max_new_tokens": 50,
        "repetition_penalty": 1.2, # Punishes the model for repeating the same words
        "no_repeat_ngram_size": 2  # Prevents 2-word phrases from looping
    }
    
    # Generate Base Caption
    base_output = base_model.generate(**inputs, **gen_kwargs)
    base_caption = processor.decode(base_output[0], skip_special_tokens=True).capitalize()
    
    # Generate Fine-Tuned Caption (if available)
    if finetuned_model is not None:
        ft_output = finetuned_model.generate(**inputs, **gen_kwargs)
        ft_caption = processor.decode(ft_output[0], skip_special_tokens=True).capitalize()
    else:
        ft_caption = base_caption
        
    # Fuse Captions
    fused_caption = fuse_captions(base_caption, ft_caption)
    
    # Backend Logging
    print("\n" + "="*40)
    print(f"🖼️  IMAGE: {os.path.basename(image_path)}")
    print(f"🔵 BASE CAPTION:       {base_caption}")
    print(f"🟢 FINE-TUNED CAPTION: {ft_caption}")
    print(f"🟣 FUSED CAPTION:      {fused_caption}")
    print("="*40 + "\n")
    
    return fused_caption

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part found in the request.')
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected for uploading.')
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image_url = url_for('static', filename='uploads/' + filename)
        
        try:
            # Generate Hybrid Caption
            caption = generate_hybrid_caption(filepath)
            
            # Generate Speech (gTTS)
            base_name = os.path.splitext(filename)[0]
            audio_filename = f"{base_name}.mp3"
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            
            if os.path.exists(audio_filepath):
                try:
                    os.remove(audio_filepath)
                except Exception as rem_err:
                    print("Could not proactively remove old audio track:", rem_err)
                    
            tts = gTTS(text=caption, lang='en')
            tts.save(audio_filepath)
            
            audio_url = url_for('static', filename='audio/' + audio_filename)
            cache_busting_url = f"{audio_url}?t={int(time.time())}"
            
            return render_template('index.html', image_url=image_url, caption=caption, audio_url=cache_busting_url)
            
        except Exception as e:
            return render_template('index.html', image_url=image_url, error=f'Processing Error: {str(e)}')

@app.route('/capture_webcam', methods=['POST'])
def capture_webcam():
    if 'image' not in request.files:
        return {'status': 'error', 'message': 'No file part'}, 400
    
    file = request.files['image']
    if file.filename == '':
        return {'status': 'error', 'message': 'No selected file'}, 400
        
    if file:
        filename = secure_filename("webcam_" + str(int(time.time())) + ".jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Generate Hybrid Caption
            caption = generate_hybrid_caption(filepath)
            
            # Generate Speech (gTTS)
            base_name = os.path.splitext(filename)[0]
            audio_filename = f"{base_name}.mp3"
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            
            if os.path.exists(audio_filepath):
                try:
                    os.remove(audio_filepath)
                except Exception as rem_err:
                    print("Could not proactively remove old audio track:", rem_err)
                    
            tts = gTTS(text=caption, lang='en')
            tts.save(audio_filepath)
            
            image_url = url_for('static', filename='uploads/' + filename)
            audio_url = url_for('static', filename='audio/' + audio_filename)
            cache_busting_url = f"{audio_url}?t={int(time.time())}"
            
            return {
                'status': 'success', 
                'filename': filename, 
                'image_url': image_url, 
                'caption': caption, 
                'audio_url': cache_busting_url
            }
        except Exception as e:
            return {'status': 'error', 'message': f'Processing Error: {str(e)}'}, 500

if __name__ == '__main__':
    app.run(debug=True)

