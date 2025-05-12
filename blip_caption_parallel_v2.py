import os
from datasets import load_from_disk, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
import multiprocessing as mp
from tqdm import tqdm

CHUNK_SIZE = 1024
NUM_PROCESSES = 4

def worker_init():
    global processor, model, device
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, torch.float16)
    model.eval()

def process_example(example):
    global processor, model, device
    image_path = example['image_path']
    
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return {"blip_caption": caption}
    except Exception:
        return {"blip_caption": None}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    save_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset"
    output_root = save_root + "_caption"
    os.makedirs(output_root, exist_ok=True)
    
    dataset = load_from_disk(save_root)
    data = dataset.to_list()
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Identify already processed chunks
    existing_chunks = {int(f.split("_")[1]) for f in os.listdir(output_root) if f.startswith("chunk_")}
    
    print(f"Skipping {len(existing_chunks)} chunks already saved.")

    with mp.get_context("spawn").Pool(processes=NUM_PROCESSES, initializer=worker_init) as pool:
        for chunk_idx in range(total_chunks):
            if chunk_idx in existing_chunks:
                continue  # Skip already processed chunks

            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(data))
            chunk_data = data[start:end]

            print(f"Processing chunk {chunk_idx} with {len(chunk_data)} samples...")

            results = list(tqdm(pool.imap(process_example, chunk_data), total=len(chunk_data)))
            merged = [{**orig, **res} for orig, res in zip(chunk_data, results)]

            chunk_ds = Dataset.from_list(merged)
            chunk_ds.save_to_disk(os.path.join(output_root, f"chunk_{chunk_idx:04d}"))

    print("All remaining chunks processed.")
