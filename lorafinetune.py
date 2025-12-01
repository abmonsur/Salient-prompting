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
# ----------------------------
# 1️⃣  Load model + tokenizer
# ----------------------------
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no pad token by default
if tokenizer.bos_token_id is None:
    # reuse EOS as BOS if needed
    tokenizer.add_special_tokens({'bos_token': tokenizer.eos_token})
    model.decoder.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# 2️⃣  Dataset setup
# ----------------------------
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

# ----------------------------
# 3️⃣  Dataset class
# ----------------------------
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, image_list, captions_dict, feature_extractor, tokenizer, max_target_length=32):
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

        # 🔹 Flatten each (image, caption) pair into a single list
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

        # 1️⃣ Open & preprocess image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {e}")

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # 2️⃣ Tokenize caption → label IDs
        labels = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        # 3️⃣ Return
        return {"pixel_values": pixel_values, "labels": labels}


train_dataset = Flickr8kDataset(image_dir, train_imgs, captions_dict, feature_extractor, tokenizer)
val_dataset = Flickr8kDataset(image_dir, val_imgs, captions_dict, feature_extractor, tokenizer)


# ----------------------------
# 4️⃣  Collator
# ----------------------------
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

# ----------------------------
# 5️⃣  Training arguments
# ----------------------------
from transformers import Seq2SeqTrainingArguments, IntervalStrategy

training_args = Seq2SeqTrainingArguments(
    output_dir="logs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    # evaluation_strategy=IntervalStrategy.EPOCH,  # or "epoch" on newer versions
    save_strategy=IntervalStrategy.EPOCH,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=100,
    learning_rate=1e-5,
    num_train_epochs=5,
    save_total_limit=2,
    report_to="none",
)


# ----------------------------
# 6️⃣  Trainer
# ----------------------------
from peft import LoraConfig, get_peft_model, TaskType

# 1️⃣ Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,              # rank of low-rank matrices
    lora_alpha=32,    # scaling
    lora_dropout=0.1,
    target_modules= "all-linear",  # GPT-2 attention layers
)

# 2️⃣ Wrap your model
model = get_peft_model(model, lora_config)

# 3️⃣ Now train with the same Seq2SeqTrainer
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

# ----------------------------
# 8️⃣  Test-set caption generation
# ----------------------------

model.eval()
results = []
max_length = 18           # slightly shorter than train captions (max_target_length=32)
num_beams = 4             # beam search width
length_penalty = 0.8      # discourage over-long sentences
early_stopping = True

for img_name in tqdm(test_imgs, desc="Generating test captions"):
    img_path = os.path.join(image_dir, img_name)

    try:
        # 1️⃣ load & preprocess
        image = Image.open(img_path).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

        # 2️⃣ generate caption
        with torch.no_grad():
            output_ids = model.base_model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        # 3️⃣ decode to text
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # 4️⃣ store result
        results.append({"image": img_name, "caption": caption})

    except Exception as e:
        print(f"⚠️  Skipping {img_name}: {e}")
        continue

test_df = pd.DataFrame(results)
test_df.to_csv("flickr8k_test_captions.csv", index=False)
print("Test captions saved to flickr8k_test_captions.csv")


# ----------------------------
# 9️⃣  Evaluation on test set
# ----------------------------
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
