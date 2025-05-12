from datasets import load_from_disk
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
import multiprocessing as mp


def generate_caption(example):
    # Each process must load its own model & processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, torch.float16)

    image_path = example['image_path']
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"blip_caption": caption}


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Avoid fork
    save_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset"
    dataset = load_from_disk(save_root)

    # Run with multiple processes; each process handles its own CUDA
    dataset = dataset.map(generate_caption, batched=False, num_proc=8)

    dataset.save_to_disk(save_root + "_caption")
    print("Done! Captions added and saved.")
