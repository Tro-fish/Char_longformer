import transformers
import wandb

task = "sts"
model_checkpoint = "klue/bert-base" # "jjp97/korean-char-longformer"
batch_size = 128
config = {
    'epochs': 5,
    'batch_size': batch_size,
}

wandb.init(project='klue', # 실험기록을 관리한 프로젝트 이름
           entity='waniboyy', # 사용자명 또는 팀 이름
           name='sts-bert-base',    # 실험에 부여한 이름               
           config=config, # 실험에 사용된 설정을 저장
          )

from datasets import load_dataset
dataset = load_dataset('klue', task) # klue 의 task=nil 을 load
# print(dataset) # dataset 구조 확인             

import torch

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#from tokenization_kocharelectra import KoCharElectraTokenizer
#tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

import datasets
import random
import pandas as pd

import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

dataset = dataset.flatten()
dataset = dataset.rename_column('labels.real-label','label')

max_length = 512

def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'],
                     truncation=True, max_length=max_length, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 1 # 유사한지 아닌지 판단하는 binary classification 문제이므로 1개의 label class

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                           num_labels=num_labels)

import os

model_name = model_checkpoint.split("/")[-1]
output_dir = os.path.join("test-klue", "sts")
logging_dir = os.path.join(output_dir, 'logs')
args = TrainingArguments(
    # checkpoint, 모델의 checkpoint 가 저장되는 위치
    output_dir=output_dir,
    # overwrite_output_dir=True,

    # Model Save & Load
    save_strategy = "epoch", # 'steps'
    load_best_model_at_end=True,
    # save_steps = 500,


    # Dataset, epoch 와 batch_size 선언
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    
    # Optimizer
    learning_rate=2e-5, # 5e-5
    weight_decay=0.01,  # 0
    # warmup_steps=200,

    # Resularization
    # max_grad_norm = 1.0,
    # label_smoothing_factor=0.1,

    # Evaluation 
    metric_for_best_model='eval_pearsonr', # task 별 평가지표 변경 
    evaluation_strategy = "epoch",

    # HuggingFace Hub Upload, 모델 포팅을 위한 인자
    # push_to_hub=True,
    # push_to_hub_model_id=f"{model_name}-finetuned-{task}",

    # Logging, log 기록을 살펴볼 위치, 본 노트북에서는 wandb 를 이용함
    logging_dir=logging_dir,
    report_to='wandb',

    # Randomness, 재현성을 위한 rs 설정
    seed=42,
)

from datasets import load_metric
# STS의 metric은 pearsonr, F1 score을 사용합니다. [TODO] F1 metric 추가
metric_pearsonr = load_metric("pearsonr") # peason r
metric_f1 = load_metric("f1") # F1

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    pr = metric_pearsonr.compute(predictions=predictions,
                                  references=labels)
    
    return pr
    #f1 = metric_f1.compute(predictions=predictions,
    #                              references=labels)
    #return {'pearsonr' : pr,
    #        'f1' : f1}

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()
trainer.evaluate()
trainer.save('/test-klue/sts/model.h5')
