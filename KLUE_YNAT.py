# 

import transformers

task = "ynat"
model_checkpoint = "jjp97/korean-char-longformer"
batch_size = 16

from datasets import load_dataset
# input -> title, target(output) -> label
dataset = load_dataset('klue','ynat') # dictionary 형태로 train, test 데이터들이 저장되어 있음.
#print(dataset) --> 데이터셋 구조 확인

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

import torch
from transformers import AutoTokenizer

from tokenization_kocharelectra import KoCharElectraTokenizer
tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

def preprocess_function(examples):
    # Tokenizer로 input_ids, attention_mask를 계산
    return tokenizer(examples["title"], padding=True, truncation=True, max_length=2048)

# 전체 데이터셋에 대해서 tokenizer를 적용
encoded_dataset = dataset.map(preprocess_function, batched=True)

# model 로드 
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
num_labels = dataset['train'].features['label'].num_classes # 7개의 label class
# Classification 문제이므로
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

import os

# trainer 객체 정의
model_name = model_checkpoint.split("/")[-1] # model_checkpoint의 이름을 가져옴
output_dir = os.path.join("test-klue", "ynat") # test-klue/ynat 경로에 저장
logging_dir = os.path.join(output_dir, 'logs') 

args = TrainingArguments(
    # checkpoint, 모델의 checkpoint 가 저장되는 위치
    output_dir=output_dir,
    overwrite_output_dir=True,

    # Model Save & Load
    save_strategy = "epoch", # 'steps'
    load_best_model_at_end=True,
    save_steps = 500,

    # Dataset, epoch 와 batch_size 선언
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    
    # Optimizer
    learning_rate=2e-5, # 5e-5
    weight_decay=0.01,  # 0
    # warmup_steps=200,

    # Resularization
    max_grad_norm = 1.0,
    label_smoothing_factor=0.1,

    # Evaluation 
    metric_for_best_model='eval_f1', # task 별 평가지표 변경
    evaluation_strategy = "epoch",

    # Logging, log 기록을 살펴볼 위치, 본 노트북에서는 wandb 를 이용함
    #logging_dir=logging_dir,
    #report_to='wandb',

    # Randomness, 재현성을 위한 rs 설정
    seed=42,
)

# dataset 라이브러리에서 제공하는 metric list 확인
from datasets import load_metric
# YNAT의 metric은 F1 score를 사용하므로 F1을 고려하여 metric 계산을 위한 함수를 정의
metric_macrof1 = load_metric('f1')
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return metric_macrof1.compute(predictions=predictions,
                                  references=labels, average='macro')

# trainer 객체 정의
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#import wandb
#id = wandb.util.generate_id()
#print(id)
#
#wandb.init(project='klue', # 실험기록을 관리한 프로젝트 이름
#           entity='waniboyy', # 사용자명 또는 팀 이름
#           id='kfy6lxfp',  # 실험에 부여된 고유 아이디
#           name='ynat-ours_v1_epoch3',    # 실험에 부여한 이름               
#          )

# Training
trainer.train()
trainer.evaluate()
trainer.save('/test-klue/char_longformer/ynat/model.h5')

#wandb.finish()