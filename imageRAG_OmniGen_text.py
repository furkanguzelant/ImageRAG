import argparse
import sys
import os
import openai
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import torch
from PIL import Image

from retrieval import *
from utils import *

def run_omnigen(prompt, input_images, out_path, args):
    print("running OmniGen inference")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1", device=device,)
    images = pipe(prompt=prompt, input_images=input_images, height=args.height, width=args.width,
                  guidance_scale=args.guidance_scale, img_guidance_scale=args.image_guidance_scale,
                  seed=args.seed, use_input_image_size_as_output=args.use_input_image_size_as_output)

    images[0].save(out_path)
    return out_path

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=20, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    return processor.decode(out[0], skip_special_tokens=True)

def retrieve_image_from_caption(captions, k=3):
  print("Captions:", captions)
  result_paths = []

  for caption in captions:
    retrieved_vectorstore = FAISS.load_local("rag_index", embedding_model, allow_dangerous_deserialization=True)

    # Example: retrieve similar captions
    results = retrieved_vectorstore.similarity_search_with_score(caption, k=k)

    # Print results with scores
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
    parser.add_argument("--omnigen_path", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.6)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--input_images", type=str, default="")
    parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation', 'personalization'])
    parser.add_argument("--model_cpu_offload", action='store_true')
    parser.add_argument("--use_input_image_size_as_output", action='store_true')
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank', 'BLIP', 'CLIP+BLIP'])
    parser.add_argument("--check_relevance", action='store_true', default=False)
    parser.add_argument("--criticize_outputs", action='store_true', default=False)
    args = parser.parse_args()

    sys.path.append(args.omnigen_path)
    from OmniGen import OmniGenPipeline

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
    input_images = args.input_images.split(",") if args.input_images else []
    k_concepts = 3 - len(input_images) if args.mode != "personalization" else 1
    k_captions_per_concept = 1

    f.write(f"prompt: {args.prompt}\n")

    blip_captions = []
    if args.retrieval_method == "BLIP":
        # Generate captions from dataset
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device, torch.float16)

        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

        dataset_path = args.dataset
        data_files = os.listdir(dataset_path) 
        for filename in os.listdir(dataset_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(dataset_path, filename)
                item = dict()
                item["image_path"] = image_path
                item["caption"] = generate_caption(image_path)
                blip_captions.append(item)


    if args.mode == "omnigen_first":
        out_name = f"{args.out_name}_no_imageRAG.png"
        out_path = os.path.join(args.out_path, out_name)
        if not os.path.exists(out_path):
            f.write(f"running OmniGen, will save results to {out_path}\n")
            run_omnigen(args.prompt, input_images, out_path, args)

        if args.only_rephrase:
            rephrased_prompt = retrieval_caption_generation(args.prompt, input_images + [out_path],
                                                            gpt_client=client,
                                                            k_captions_per_concept=k_captions_per_concept,
                                                            only_rephrase=args.only_rephrase)
            if rephrased_prompt == True:
                f.write("result matches prompt, not running imageRAG.")
                f.close()
                exit()

            f.write(f"running OmniGen, rephrased prompt is: {rephrased_prompt}\n")
            out_name = f"{args.out_name}_rephrased.png"
            out_path = os.path.join(args.out_path, out_name)
            run_omnigen(rephrased_prompt, input_images, out_path, args)
            f.close()
            exit()
        else:
            ans = retrieval_caption_generation(args.prompt,
                                               input_images + [out_path],
                                               gpt_client=client,
                                               k_captions_per_concept=k_captions_per_concept)

            if type(ans) != bool:
                captions = convert_res_to_captions(ans)
                f.write(f"captions: {captions}\n")
            else:
                f.write("result matches prompt, not running imageRAG.")
                f.close()
                exit()

        omnigen_out_path = out_path

    elif args.mode == "generation":
        captions = retrieval_caption_generation(args.prompt,
                                                input_images,
                                                gpt_client=client,
                                                k_captions_per_concept=k_captions_per_concept,
                                                decision=False)
        captions = convert_res_to_captions(captions)
        f.write(f"captions: {captions}\n")

    def retrieve_image(caption, k, retrieval_method, blip_captions):
      if retrieval_method == "BLIP":
          create_embeddings(blip_captions)
          paths = retrieve_image_from_caption(caption, k=k)
      elif retrieval_method == "CLIP+BLIP":
          paths = retrieve_img_per_caption(caption, retrieval_image_paths, embeddings_path=embeddings_path,
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
          paths = retrieve_img_per_caption(caption, retrieval_image_paths, embeddings_path=embeddings_path,
                                          k=k, device=device, method=args.retrieval_method)
      return paths
      
    k_imgs_per_caption = 1
    paths = retrieve_image(captions, k=k_imgs_per_caption, retrieval_method=args.retrieval_method, blip_captions=blip_captions)
    final_paths = np.array(paths).flatten().tolist()
    j = len(input_images)
    k = 3  # can use up to 3 images in prompt with omnigen
    paths = final_paths[:k - j]
    f.write(f"final retrieved paths: {paths}\n")
    image_paths_extended = input_images + paths

    examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(paths))])
    prompt_w_retreival = f"According to these images of {examples}, generate {args.prompt}"
    f.write(f"prompt_w_retreival: {prompt_w_retreival}\n")

    out_name = f"{args.out_name}_gs_{args.guidance_scale}_im_gs_{args.image_guidance_scale}.png"
    out_path = os.path.join(args.out_path, out_name)
    f.write(f"running OmniGen, will save result to: {out_path}\n")

    out_image_paths = run_omnigen(prompt_w_retreival, image_paths_extended, out_path, args)

    if args.criticize_outputs:
      print("Criticizing output")
      best_img_path, max_score = rate_generated_outputs(args.prompt, [out_image_paths], client)

      while max_score < 4:
        print("Output image does not match to the prompt")
        ans = retrieval_caption_generation(args.prompt,
                                               input_images + [out_image_paths],
                                               gpt_client=client,
                                               k_captions_per_concept=k_captions_per_concept)
        if type(ans) != bool:
            captions = convert_res_to_captions(ans)
            f.write(f"captions: {captions}\n")

        # else use the previous generated captions

        input_images = [out_image_paths] if max_score > 1 else []
        paths = retrieve_image(captions, k=k_imgs_per_caption, retrieval_method=args.retrieval_method, blip_captions=blip_captions)
        final_paths = np.array(paths).flatten().tolist()
        j = len(input_images)
        k = 3  # can use up to 3 images in prompt with omnigen
        paths = final_paths[:k - j]
        f.write(f"final retrieved paths: {paths}\n")
        image_paths_extended = input_images + paths

        previous_img_desc = f"Generated base image: <img><|image_{j}|></img>"
        concept_examples = ", ".join([f'{captions[i]}: <img><|image_{i + j + 1}|></img>' for i in range(len(captions))])
        prompt_w_retreival =  f"Based on the previous {previous_img_desc}, and the following examples {concept_examples}, generate {args.prompt}"

        print(prompt_w_retreival)
        out_image_paths = run_omnigen(prompt_w_retreival, image_paths_extended, out_path, args)

        best_img_path, max_score = rate_generated_outputs(args.prompt, [out_image_paths], client)
        # maybe we can ask VLM to find the issue in the generated image
        # TODO

      f.close()
      out_path = os.path.join(args.out_path, args.out_name)
      new_path = out_path + "_final." + best_img_path.split(".")[-1]
      os.rename(best_img_path, new_path)
      print("Generated image path: ", new_path)

    exit()