import argparse
import os
from os.path import isdir
from PIL import Image
from diffusers.pipelines import kandinsky3
import numpy as np
import openai
from pydantic_core.core_schema import dataclass_args_schema
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import requests
from utils import *
from retrieval import *

def generate_caption(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device, torch.float16)

    try:
      image = Image.open(image_path).convert("RGB")
      inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
      out = model.generate(**inputs, max_new_tokens=20, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
      return processor.decode(out[0], skip_special_tokens=True)
    except:
      print("Cannot process image ", image_path)
      return ""

def retrieve_image_from_caption(caption, k=3, index_dir="rag_index"):
  print("Caption:", caption)
  retrieved_vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)

  # Example: retrieve similar captions
  results = retrieved_vectorstore.similarity_search_with_score(caption, k=k)

  # Print results with scores
  result_paths = []
  for i, (doc, score) in enumerate(results):
      print(f"{i+1}. Caption: {doc.page_content}")
      print(f"   Image Url: {doc.metadata['image_url']}")
      print(f"   Similarity Score (lower = more similar): {score:.4f}")

      image_url = doc.metadata["image_url"]
      os.makedirs("retrieved_imgs", exist_ok=True)
      image_path = f"retrieved_imgs/img_ref_{i+1}.jpg"

      if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(image_path, "wb") as file:
                file.write(response.content)
            print(f"   Downloaded: {image_path}")
        except requests.RequestException as e:
            print(f"   Failed to download {image_url}: {e}")
            image_path = ""  # Reset the path if the download fails

        result_paths.append(image_path)

  return result_paths

def create_embeddings(captions):
    documents = [
        Document(page_content=item["caption"], metadata={"image_url": item["image_url"]})
        for item in captions
    ]
    # Step 3: Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(documents, embedding_model)
    # Step 4: Save the FAISS index
    os.makedirs("rag_index", exist_ok=True)
    vectorstore.save_local("rag_index")

def generate_images_with_reference(images, new_prompt, out_path, out_name):

    if type(images) is not list:
      images = [images]
    
    outs = []
    for idx, image_path in enumerate(images):
      image = Image.open(image_path)

      out_image = pipe_ip(
          prompt=new_prompt,
          ip_adapter_image=image,
          negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
          num_inference_steps=50,
          generator=generator2,
      ).images[0]
    
      cur_out_path = os.path.join(out_path, f"{out_name}_{idx}.png")
      outs.append(cur_out_path)
      out_image.save(cur_out_path)

    return outs

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
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank', 'BLIP', 'CLIP+BLIP'])
    parser.add_argument("--check_relevance", action='store_true', default=False)
    parser.add_argument("--criticize_outputs", action='store_true', default=False)

    args = parser.parse_args()

    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    client = openai.OpenAI()

    os.makedirs(args.out_path, exist_ok=True)
    out_txt_file = os.path.join(args.out_path, args.out_name + ".txt")
    f = open(out_txt_file, "w")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    data_path = args.dataset

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

    captions = []
    if args.retrieval_method == "BLIP" or args.retrieval_method == "CLIP+BLIP":
        # Generate captions from dataset
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

        with open('laion_data/captions.json', 'r') as file:
          caption_data = json.load(file)

        for data in caption_data:
            image_url = data["url"]
            item = dict()
            item["image_url"] = image_url
            item["caption"] = data["blip_caption"]
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

    def retrieve_image(caption, k, retrieval_method, blip_captions):
      if retrieval_method == "BLIP":
          if os.path.isdir("laion_data/rag_index"):
            paths = retrieve_image_from_caption(caption, k=k, index_dir="laion_data/rag_index")
          else:
            create_embeddings(blip_captions)
            paths = retrieve_image_from_caption(caption, k=k)
      elif retrieval_method == "CLIP+BLIP":
          paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                          k=100, device=device, method="CLIP")
          captions = []
          paths = paths[0]
          for image_path in paths:
              item = dict()
              item["image_path"] = image_path
              item["caption"] = generate_caption(image_path)
              captions.append(item)

          create_embeddings(blip_captions)

          paths = retrieve_image_from_caption(caption, k=k)
      else:
          paths = retrieve_img_per_caption([caption], retrieval_image_paths, embeddings_path=embeddings_path,
                                          k=k, device=device, method=args.retrieval_method)
      return paths

    paths = retrieve_image(caption, k=3, retrieval_method=args.retrieval_method, blip_captions=captions)

    if args.check_relevance:
      if args.retrieval_method != "BLIP" and args.retrieval_method != "CLIP+BLIP":
        paths = paths[0]
      relevance_score = 0
      max_attempts = 3
      attempts = 0
      while relevance_score == 0 and attempts < max_attempts:
        relevance = check_retrieved_image_relevance(caption, paths, client)
        # Sort it based on relevance 
        relevance_results = dict(sorted(relevance.items(), key=lambda item: item[1], reverse=True))
        print("Relevance: ", relevance)

        paths = list(relevance_results.keys())
        best_path = paths[0]
        relevance_score = relevance_results[best_path]

        if relevance_score == 0:
          print("Retrieved images are not relevant")
          attempts += 1
          k = len(paths) * 2
          paths = retrieve_image(caption, k=k, retrieval_method=args.retrieval_method, blip_captions=captions)
          paths = paths[k // 2:]
          print(paths)
          # TODO 
          # maybe change retrieval method?          

    if args.criticize_outputs:
      image_path = paths
    else:
      image_path = paths[0].item(0)
    print("ref path:", image_path)

    new_prompt = f"According to this image of {caption}, generate {args.prompt}"

    print("New prompt: ", new_prompt)
    out_image_paths = generate_images_with_reference(image_path, new_prompt, args.out_path, args.out_name)

    if args.criticize_outputs:
      print("Criticizing output")
      best_img_path, max_score = rate_generated_outputs(args.prompt, out_image_paths, client)

      while max_score < 3:
        print("Output image does not match to the prompt")
        ans = retrieval_caption_generation(new_prompt, [best_img_path],
                                   gpt_client=client,
                                   k_captions_per_concept=1,
                                   k_concepts=1,
                                   only_rephrase=args.only_rephrase)
        caption = ans
        caption = convert_res_to_captions(caption)[0]

        paths = retrieve_image(caption, k=3, retrieval_method=args.retrieval_method, blip_captions=captions)
        new_prompt = f"According to this image of {caption}, generate {args.prompt}"
        out_image_paths = generate_images_with_reference(paths, new_prompt, args.out_path, args.out_name)

        best_img_path, max_score = rate_generated_outputs(args.prompt, out_image_paths, client)
        # maybe we can ask VLM to find the issue in the generated image
        # TODO

      out_path = os.path.join(args.out_path, args.out_name)
      new_path = out_path + "_final." + best_img_path.split(".")[1]
      os.rename(best_img_path, new_path)
