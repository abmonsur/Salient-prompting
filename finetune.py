import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
from collections import defaultdict
from random import choice

model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({'bos_token': tokenizer.eos_token})
    model.decoder.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_dir = "Flicker8k_Dataset"
caption_file = "Flickr8k.token.txt"
train_split = "Flickr_8k.trainImages.txt"
val_split = "Flickr_8k.devImages.txt"
test_split = "Flickr_8k.testImages.txt"

# Parse captions file
captions_dict = {}
with open(caption_file, "r") as f:
    for line in f:
        img, caption = line.strip().split("\t")
        img = img.split("#")[0]
        captions_dict.setdefault(img, []).append(caption)

def load_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]

train_imgs = load_split(train_split)
val_imgs = load_split(val_split)
test_imgs = load_split(test_split)


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, image_list, captions_dict, feature_extractor, tokenizer, max_target_length=32):
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

        self.pairs = []
        for img_name in image_list:
            if img_name in captions_dict:
                for caption in captions_dict[img_name]:
                    self.pairs.append((img_name, caption))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, caption = self.pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {e}")

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


train_dataset = Flickr8kDataset(image_dir, train_imgs, captions_dict, feature_extractor, tokenizer)
val_dataset = Flickr8kDataset(image_dir, val_imgs, captions_dict, feature_extractor, tokenizer)


@dataclass
class DataCollatorForVisionEncoderDecoder:
    feature_extractor: ViTImageProcessor
    tokenizer: AutoTokenizer
    pad_token_id: int = tokenizer.pad_token_id

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = [f["labels"] for f in features]
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        labels[labels == self.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

data_collator = DataCollatorForVisionEncoderDecoder(feature_extractor, tokenizer)

from transformers import Seq2SeqTrainingArguments, IntervalStrategy

training_args = Seq2SeqTrainingArguments(
    output_dir="logs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    save_strategy=IntervalStrategy.EPOCH,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=100,
    learning_rate=1e-5,
    num_train_epochs=5,
    save_total_limit=2,
    report_to="none",
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model("./vitgpt2-flickr8k-final")
print("Fine-tuning completed and model saved!")


model.eval()
results = []
max_length = 18           
num_beams = 4             
length_penalty = 0.8      
early_stopping = True

for img_name in tqdm(test_imgs, desc="Generating test captions"):
    img_path = os.path.join(image_dir, img_name)

    try:
        image = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        results.append({"image": img_name, "caption": caption})

    except Exception as e:
        print(f"⚠️  Skipping {img_name}: {e}")
        continue

test_df = pd.DataFrame(results)
test_df.to_csv("flickr8k_test_captions.csv", index=False)
print("Test captions saved to flickr8k_test_captions.csv")


bleu = evaluate.load("bleu")
# meteor = evaluate.load("meteor")
# rouge = evaluate.load("rouge")
# cider = evaluate.load("cider")

# Load ground-truth captions for test images
gt_captions = defaultdict(list)
with open(caption_file, "r") as f:
    for line in f:
        img, caption = line.strip().split("\t")
        img = img.split("#")[0]
        if img in test_imgs:
            gt_captions[img].append(caption.strip())

predictions, references = [], []
for _, row in test_df.iterrows():
    img = row["image"]
    if img in gt_captions:
        predictions.append(row["caption"])
        references.append(gt_captions[img])

print("\nSample predictions vs refs:")
for i in range(3):
    print("PRED:", predictions[i])
    print("REFS:", references[i][:2])
    print("-" * 60)

bleu_score = bleu.compute(predictions=predictions, references=references)

print(f"BLEU:   {bleu_score['bleu']:.4f}")
