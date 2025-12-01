import os
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from random import sample
from collections import defaultdict
import evaluate

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)


image_dir = "Flicker8k_Dataset"
caption_file = "Flickr8k.token.txt"
train_split = "Flickr_8k.trainImages.txt"
test_split  = "Flickr_8k.testImages.txt"

def load_list(path):
    with open(path) as f:
        return [x.strip() for x in f]

train_imgs = load_list(train_split)
test_imgs  = load_list(test_split)

captions_dict = defaultdict(list)
with open(caption_file) as f:
    for line in f:
        img, caption = line.strip().split("\t")
        img = img.split("#")[0]
        captions_dict[img].append(caption)



def load_image(path):
    return Image.open(path).convert("RGB")



def build_fewshot_messages(fewshot_pairs, query_image_path):

    messages = []

    for img_path, cap in fewshot_pairs:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": (
                        "Here is an example image.\n"
                        "Please generate a **saliency-focused caption**. "
                        "Describe the parts of the scene that draw **immediate human attention**—"
                        "typically the largest, closest, or most visually dominant objects. "
                        "Ignore or minimize background details."
                    )
                },
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{cap}\n"
                        "(This caption focuses on the most salient objects.)"
                    )
                }
            ],
        })


    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": query_image_path},
            {
                "type": "text",
                "text": (
                    "Generate a **saliency-based caption** for this image. "
                    "Describe what a human would notice **first**, focusing on the most visually "
                    "dominant or important objects. Keep the caption concise and prioritize salient items."
                )
            },
        ],
    })

    return messages


def qwen_generate_caption(messages):
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80)

    trimmed_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


K = 18   

results = []

for img_name in tqdm(test_imgs, desc="Qwen2 Few-Shot Captioning"):
    query_path = os.path.join(image_dir, img_name)

    examples = sample(train_imgs, K)
    fewshot_pairs = [
        (os.path.join(image_dir, x), captions_dict[x][0])
        for x in examples
    ]

    messages = build_fewshot_messages(fewshot_pairs, query_path)

    pred_caption = qwen_generate_caption(messages)

    results.append({"image": img_name, "caption": pred_caption})


df = pd.DataFrame(results)
df.to_csv("qwen2vl_2B_fewshot_flickr8k_predictions.csv", index=False)
print("\nSaved → qwen2vl_2B_fewshot_flickr8k_predictions.csv")



bleu = evaluate.load("bleu")

preds = df["caption"].tolist()
refs  = [captions_dict[row["image"]] for _, row in df.iterrows()]

bleu_score = bleu.compute(predictions=preds, references=refs)

print("\nBLEU Score:", bleu_score["bleu"])

print("\nSample Predictions:")
for i in range(5):
    print("IMG:", df.iloc[i]["image"])
    print("PRED:", df.iloc[i]["caption"])
    print("REF:", refs[i][:2])
    print("-" * 60)
