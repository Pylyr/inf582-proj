from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.tokenization_utils import BatchEncoding
import pandas as pd
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
import time

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model.to(device)

# train_df = pd.read_csv('data/train.csv', dtype={'text': str, 'titles': str})
validation_df = pd.read_csv('data/validation.csv', dtype={'text': str, 'titles': str})
# longest title is 967 words

def tokenize_text(text: pd.Series) -> BatchEncoding:
    tokens = tokenizer(
        text.tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    return tokens


def mbart_summary(input_ids, attention_mask, batch_size: int = 4) -> list:
    summaries = []
    tokenizer.src_lang = "fr_XX"
    assert isinstance(model, MBartForConditionalGeneration)

    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(device)

        summary_tokens = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_tokens]
        summaries.extend(batch_summaries)

    return summaries

def score_summaries(predicted_summary: pd.Series, reference_summary: pd.Series):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for i in tqdm(range(len(predicted_summary))):
        score = scorer.score(predicted_summary[i], reference_summary[i])[
            'rougeL'][2]
        scores.append(score)
    avg_score = sum(scores) / len(scores)

    return avg_score



validation_tensor = tokenize_text(validation_df['text'][:64])

# bath_size = 32 crashes the machine

summaries = mbart_summary(validation_tensor['input_ids'], validation_tensor['attention_mask'], batch_size=16)