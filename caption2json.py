from datasets import load_from_disk, load_dataset
import os
import json

dataset_main = "laion/laion400m"
data_main = load_dataset(dataset_main, split="train[:10%]")

# Directory containing chunked datasets
chunk_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset_caption_full"

# Output JSON path
output_json_path = os.path.join(chunk_root, "captions.json")

captions_list = []

# Extract captions from each chunk
dataset = load_from_disk(chunk_root)
for example in dataset:
    file_name = example["image_path"].split("/")[-1]
    filenum = file_name.split(".")[0]
    print(f"Processing {filenum}...")

    data = data_main[int(filenum)]
    url = data["url"]
    if example.get("blip_caption") is not None:
        captions_list.append({
            "url": url,
            "image_path": example["image_path"],
            "blip_caption": example["blip_caption"],
            "caption": example["caption"],
        })

# Write to JSON
with open(output_json_path, "w") as f:
    json.dump(captions_list, f, indent=2)

print(f"Saved {len(captions_list)} captions to {output_json_path}")
