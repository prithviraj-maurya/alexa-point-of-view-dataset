# ### Converting Point of View for Virtual Assistants using Huggingface transformers

## Imports
#!pip install transformers[torch] datasets sacrebleu evaluate accelerate --quiet
from huggingface_hub import notebook_login
import random
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import spacy
import accelerate
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, pipeline, Seq2SeqTrainer


# set options
pd.set_option('max_colwidth', None) # show full text
random.seed(42)
metric = evaluate.load("sacrebleu") #https://www.youtube.com/watch?v=M05L1DhFqcw


## Read Data

train_df = pd.read_csv(f"/content/data/train.tsv", sep="\t", dtype={"input": str, "output": str})
test_df = pd.read_csv(f"/content/data/test.tsv", sep="\t", dtype={"input": str, "output": str})
dev_df = pd.read_csv(f"/content/data/dev.tsv", sep="\t", dtype={"input": str, "output": str})
total_df = pd.read_csv(f"/content/data/total.tsv", sep="\t", dtype={"input": str, "output": str})
print("Train", train_df.shape)
print("Test", test_df.shape)
print("Dev", dev_df.shape)
print("Total", total_df.shape)
train_df.head()

## Configs
MAX_LEN = 128

## Utility functions
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def get_response(input_text,num_return_sequences=1,num_beams=1):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=MAX_LEN, return_tensors="pt")
  model.to("cpu")
  translated = model.generate(**batch,max_length=MAX_LEN,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

## Create dataset


raw_datasets = load_dataset("/content/data/")
print(raw_datasets)

## Tokenizer
model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt", model_max_length=MAX_LEN)

## Tokenize the dataset

def tokenize_function(example):
  return tokenizer(example["input"], truncation=True, text_target=example["output"], max_length=MAX_LEN)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["input", "output"])
print(tokenized_datasets)

## Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

## Training

args = Seq2SeqTrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    output_dir="/output/"
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

## Check BLEU score on test
trainer.evaluate(eval_dataset=tokenized_datasets["test"])

translated = trainer.predict(tokenized_datasets["test"], max_length=20)

test_df["predicted"] = tokenizer.batch_decode(translated[0], skip_special_tokens=True)

print(test_df.head())

test_df.input.apply(len).min(), test_df.input.apply(len).mean(), test_df.input.apply(len).max()

test_df.predicted.apply(len).min(), test_df.predicted.apply(len).mean(), test_df.predicted.apply(len).max()

test_df["bleu_score"] = test_df.apply(lambda row: metric.compute(predictions=[row["predicted"]], references=[[row["output"]]])["score"], axis=1)

print(test_df.head())

test_df.to_csv("test_converted_t5_HF.csv", index=False)


# ### Examples that went wrong
train_processed = test_df.copy()
train_processed_sorted = train_processed.sort_values("bleu_score", ascending=False)

print(train_processed_sorted.tail(20))
