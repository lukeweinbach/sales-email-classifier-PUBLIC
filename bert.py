# IMPLEMENTATION OF THE MODEL TRAINING AND TESTING
# Author: Luke Weinbach

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np

device = torch.device('mps')


###########################
## CREATING THE DATASETS ##
########################### 




# Initialize all variables we will need for training
data_path = "clean-data-final.csv" 
text_column_name = "Body" 
label_column_name = "Category" 
model_name = "distilbert/distilbert-base-uncased"
test_size = 0.2 
num_labels = 3 
max_length = 64 # DEFAULT - 156

print("\n******************")
print(f"MAX LENGTH: {max_length}")
print("******************\n")

# Read in the data from the path
df = pd.read_csv(data_path)

# Setup for Data Preprocessing
le = preprocessing.LabelEncoder()
le.fit(df[label_column_name].tolist())
df['label'] = le.transform(df[label_column_name].tolist())
df_train, df_test = train_test_split(df, test_size=test_size)

tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def preprocess_function(examples):
    return tokenizer(examples[text_column_name], max_length=max_length, truncation=True)

# Initialize the training and test sets as a Dataset objects
train_dataset = Dataset.from_pandas(df_train)
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.remove_columns([text_column_name, label_column_name])
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_train = tokenized_train.with_format("torch")

test_dataset = Dataset.from_pandas(df_test)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
tokenized_test = tokenized_test.remove_columns([text_column_name])
tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test = tokenized_test.with_format("torch")


########################
## TRAINING THE MODEL ##
######################## 

# Conversion dictionaries between the class id numbers and their corresponding labels
id2label = {
    0 : "DOES NOT REQUIRE ATTENTION",
    1 : "REQUIRES EVENTUAL ATTENTION",
    2 : "REQUIRES IMMEDIATE ATTENTION"
    }
label2id = {
    "DOES NOT REQUIRE ATTENTION"    : 0,
    "REQUIRES EVENTUAL ATTENTION"   : 1,
    "REQUIRES IMMEDIATE ATTENTION"  : 2
    }

# Initialize DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)


# Define the evaluation function
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Set the training arguments
training_args = TrainingArguments(
    output_dir="checkpoint",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy = "epoch",
    logging_strategy="epoch"
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics 
)


###################
## TRAINING STEP ##
################### 


trainer.train()


###############
# NOTE Uncomment the code below to save an iteration of the model
# NOTE 'save_path' should be a path to an empty directory. the empty directory 'save_model_here' is provided in this directory by default.
# NOTE use 'mkdir [New Directory Name]' in the command line to create a new directory and replace "save_model_here" in save_path below 
# with the new directory's name that you want to be the destination for the model save
###############
# save_path = "save_model_here"
# trainer.save_model(save_path)