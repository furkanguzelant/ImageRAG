from datasets import load_from_disk, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
import multiprocessing as mp
from functools import partial
import os

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
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {"blip_caption": None}
    
    try: 
        inputs = processor(images=[image], return_tensors="pt").to(device, torch.float16)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {"blip_caption": None}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)

    
    return {"blip_caption": caption}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    save_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset"
    dataset = load_from_disk(save_root)

    data = dataset.to_list()

    print("Starting pool...")
    with mp.get_context("spawn").Pool(processes=4, initializer=worker_init) as pool:
        results = list(pool.imap(process_example, data))

    # Combine results back into a dataset
    updated_dataset = Dataset.from_list([
        {**orig, **res} for orig, res in zip(data, results)
    ])

    updated_dataset.save_to_disk(save_root + "_caption")
    print("Done!")
