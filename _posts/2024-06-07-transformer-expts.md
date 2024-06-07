---
layout: post
title: Transformer experiments - DistilBERT
---

In this post, we will experiment with DistilBERT - a transformer based language model available on HuggingFace.

We start by installing the libraries recommended on HuggingFace's [quick tour](https://huggingface.co/docs/transformers/en/quicktour).


```python
!pip install transformers datasets evaluate accelerate
```


```python
!pip install torch
```

DistilBERT is an efficient version of the larger BERT model. Let us start by testing DistilBERT for masked language modeling (needs `DistilBertForMaskedLM`).


```python
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

input_text = "The aliens are eating [MASK] happily. Their favorite hobby is [MASK]."
encoded_input = tokenizer(input_text, return_tensors='pt')

output = model(**encoded_input)
predicted_token_id = output.logits.argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id[0])

print("Predicted token:", predicted_token)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(


    Predicted token: . the aliens are eating fruit happily. their favorite hobby is cooking..


Interesting response! Let us now try using DistilBERT for a text classification task (needs `DistilBertForSequenceClassification`) on the imdb movie reviews dataset. We start by loading the dataset, and then tokenizing it using the same tokenizer (`DistilBertTokenizer`) as earlier. The latter can take up to 12 mins on Google Colab.


```python
from datasets import load_dataset
```


```python
dataset = load_dataset("imdb")
```


```python
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize, batched=True)
```


```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
```


```python
model2 = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
training_args = TrainingArguments(
    output_dir="imdb_output",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    fp16=True,
    num_train_epochs=2,
)

trainer = Trainer(
    model=model2,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

```

Unfortunately, with just 12.67 GB free RAM on Google Colab, the above training job crashes the moment it uses up all the available memory (about a minute or two). This means we cannot progress with the IMDB reviews sentiment classification task at the moment. We hope we can complete this task with better compute access in the future. Nevertheless, we got a decent first exposure to the HuggingFace ecosystem using the DistilBERT model for masked language modelling, and using its tokenizer to tokenize the IMDB movie reviews dataset.
