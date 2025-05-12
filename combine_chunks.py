from datasets import Dataset, concatenate_datasets, load_from_disk
import os

# Path where chunked datasets are saved
output_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset_caption"

# List and sort chunk paths
chunk_dirs = sorted([
    os.path.join(output_root, d)
    for d in os.listdir(output_root)
    if d.startswith("chunk_")
])

# Load all chunks
all_chunks = [load_from_disk(chunk_path) for chunk_path in chunk_dirs]

# Combine
full_dataset = concatenate_datasets(all_chunks)

# Optionally save the combined dataset
combined_output_path = output_root + "_full"
full_dataset.save_to_disk(combined_output_path)

print(f"Combined dataset saved to: {combined_output_path}")
