import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from utils import *
from retrieval import *

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=20, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    return processor.decode(out[0], skip_special_tokens=True)

def retrieve_image_from_caption(caption, k=3):
  print("Caption:", caption)
  retrieved_vectorstore = FAISS.load_local("rag_index", embedding_model, allow_dangerous_deserialization=True)

  # Example: retrieve similar captions
  query = "cradle"
  results = retrieved_vectorstore.similarity_search_with_score(query, k=k)

  # Print results with scores
  result_paths = []
  for i, (doc, score) in enumerate(results):
      print(f"{i+1}. Caption: {doc.page_content}")
      print(f"   Image Path: {doc.metadata['image_path']}")
      print(f"   Similarity Score (lower = more similar): {score:.4f}")
      result_paths.append(doc.metadata["image_path"])

  return result_paths

def create_embeddings(captions):
    documents = [
        Document(page_content=item["caption"], metadata={"image_path": item["image_path"]})
        for item in captions
    ]
    # Step 3: Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(documents, embedding_model)
    # Step 4: Save the FAISS index
    os.makedirs("rag_index", exist_ok=True)
    vectorstore.save_local("rag_index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=0.5)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])

    args = parser.parse_args()

    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    client = openai.OpenAI()

    os.makedirs(args.out_path, exist_ok=True)
    out_txt_file = os.path.join(args.out_path, args.out_name + ".txt")
    f = open(out_txt_file, "w")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    data_path = f"datasets/{args.dataset}"

    prompt_w_retreival = args.prompt

    retrieval_image_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]

    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset}"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    )

    pipe_clean = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    generator1 = torch.Generator(device="cuda").manual_seed(args.seed)
    pipe_ip = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    pipe_ip.load_ip_adapter("h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                            cache_dir=args.hf_cache_dir)

    pipe_ip.set_ip_adapter_scale(args.ip_scale)
    generator2 = torch.Generator(device=device).manual_seed(args.seed)

    sd_first = args.mode == "sd_first"

    
    # Generate captions from dataset
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, torch.float16)

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

    dataset_path = f"datasets/{args.dataset}"
    captions = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
            item = dict()
            item["image_path"] = image_path
            item["caption"] = generate_caption(image_path)
            captions.append(item)


    if sd_first:
        cur_out_path = os.path.join(args.out_path, f"{args.out_name}_no_imageRAG.png")
        if not os.path.exists(cur_out_path):
            out_image = pipe_clean(
                prompt=args.prompt,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=50,
                generator=generator1,
            ).images[0]
            out_image.save(cur_out_path)

        ans = retrieval_caption_generation(args.prompt, [cur_out_path],
                                           gpt_client=client,
                                           k_captions_per_concept=1,
                                           k_concepts=1,
                                           only_rephrase=args.only_rephrase)
        if type(ans) != bool:
            if args.only_rephrase:
                print(f"running SDXL, rephrased prompt is: {ans}\n")
                cur_out_path = os.path.join(args.out_path, f"{args.out_name}_rephrased.png")
                out_image = pipe_clean(
                    prompt=ans,
                    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                    num_inference_steps=50,
                    generator=generator1,
                ).images[0]
                out_image.save(cur_out_path)
                exit()

            caption = ans
            caption = convert_res_to_captions(caption)[0]
            print(f"caption: {caption}\n")
        else:
            print(f"prompt: {args.prompt}")
            print("result matches prompt, not running imageRAG.")
            exit()
    else:
        caption = retrieval_caption_generation(args.prompt, [],
                                               gpt_client=client,
                                               k_captions_per_concept=1,
                                               decision=False)
        caption = convert_res_to_captions(caption)[0]
        f.write(f"captions: {caption}\n")

    if args.retrieval_method == "BLIP":
        paths = retrieve_image_from_caption(caption, k=1)
    elif args.retrieval_method == "CLIP+BLIP":
        paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                        k=100, device=device, method="CLIP")
        captions = []
        paths = paths[0]
        for image_path in paths:
            item = dict()
            item["image_path"] = image_path
            item["caption"] = generate_caption(image_path)
            captions.append(item)

        create_embeddings(captions)

        paths = retrieve_image_from_caption(caption, k=1)
    else:
        paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                        k=1, device=device, method=args.retrieval_method)
    
    image_path = np.array(paths).flatten()[0]
    print("ref path:", image_path)

    new_prompt = f"According to this image of {caption}, generate {args.prompt}"
    image = Image.open(image_path)

    out_image = pipe_ip(
        prompt=new_prompt,
        ip_adapter_image=image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=50,
        generator=generator2,
    ).images[0]

    cur_out_path = os.path.join(args.out_path, f"{args.out_name}.png")
    out_image.save(cur_out_path)