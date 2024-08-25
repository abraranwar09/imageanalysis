import os
import time
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, UnidentifiedImageError
import torch

# Initialize the model and tokenizer with the Hugging Face token
model_id = "vikhyatk/moondream2"
revision = "2024-07-23"
hf_token = "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"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=hf_token)

# Database setup
def get_db_connection():
    conn = sqlite3.connect('image_descriptions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS descriptions (filename TEXT, description TEXT)''')
    conn.commit()
    return conn, c

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            self.process_image(event.src_path)

    def process_image(self, image_path):
        # Create a new database connection for this thread
        conn, c = get_db_connection()
        
        try:
            image = Image.open(image_path)
            enc_image = model.encode_image(image)
            
            # Generate attention mask
            inputs = tokenizer("Describe this image.", return_tensors="pt")
            attention_mask = inputs['attention_mask']
            
            # Generate description
            description_ids = model.generate(enc_image, attention_mask=attention_mask)
            description = tokenizer.decode(description_ids[0], skip_special_tokens=True)
            
            # Print the description
            print(f"Description for {image_path}: {description}")
            
            # Save to database
            c.execute("INSERT INTO descriptions (filename, description) VALUES (?, ?)", (os.path.basename(image_path), description))
            conn.commit()
            print(f"Processed and saved description for {image_path}")
        except (UnidentifiedImageError, SyntaxError) as e:
            print(f"Failed to process {image_path}: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    path = "/mnt/flaskserver/demozone/demozone/uploads/images"  # Updated path
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"Monitoring {path} for new images...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()