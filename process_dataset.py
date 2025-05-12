import os
from datasets import load_dataset, Dataset
from tqdm import tqdm

last_index = 489601  # Number of samples you already downloaded

def create_subset(dataset_name, save_root="./subset_dataset"):
    dataset = load_dataset(dataset_name, split="train")

    small_dataset = dataset.select(range(last_index))

    images_folder = os.path.join(save_root, "images")
    os.makedirs(images_folder, exist_ok=True)

    metadata = []  # Store the other fields per sample
    for idx, sample in tqdm(enumerate(small_dataset), total=len(small_dataset), desc="Processing metadata"):
        sample_metadata = {k: v for k, v in sample.items() if k != 'image' and k != 'url'}
        metadata.append(sample_metadata)

    new_data = {k: [] for k in metadata[0].keys()}
    new_data["image_path"] = []

    missing_count = 0

    for idx, sample_meta in tqdm(enumerate(metadata), total=len(metadata), desc="Combining images and metadata"):
        filename = f"{idx}.jpg"
        file_path = os.path.join(images_folder, filename)

        if os.path.exists(file_path):
            # Add metadata fields
            for key, value in sample_meta.items():
                new_data[key].append(value)
            # Add image path
            new_data["image_path"].append(file_path)
        else:
            missing_count += 1  # Image is missing, skip this sample

    print(f"Missing images: {missing_count}")

    new_dataset = Dataset.from_dict(new_data)
    new_dataset.save_to_disk(save_root)

    print(f"Subset dataset saved to {save_root}")

if __name__ == "__main__":
    dataset_name = "laion/laion400m"
    save_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset"
    create_subset(dataset_name, save_root)
