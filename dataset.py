from datasets import Dataset

# Path where you saved your combined dataset
save_root = "/leonardo_work/EUHPC_A02_031/ImageRAG/datasets/laion_dataset_caption_full"

# Load it
dataset = Dataset.load_from_disk(save_root)

# Now you can use it
print(dataset[1023]) 