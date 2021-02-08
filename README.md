# SpanBERT for Web Documents
This repository is a fork from [SpanBERT](https://github.com/facebookresearch/SpanBERT) by Facebook Research, which contains code and models for the paper: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).

We have adapted the SpanBERT scripts to support general documents not in the TACRED dataset. 

## Install Requirements

```bash
pip install -r requirements.txt
```

## Download Pre-Trained SpanBERT (Fine-Tuned in TACRED)
SpanBERT has the same model configuration as [BERT](https://github.com/google-research/bert) but it differs in
both the masking scheme and the training objectives.

* [SpanBERT (large & cased)](https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz): 24-layer, 1024-hidden, 16-heads, 340M parameters

To download the SpanBERT model that was fine-tuned for relation classification in TACRED run: 

```bash
./download_finetuned.sh
```

## Apply SpanBERT 
Input is a list of python dicts, where each dict contains the sentence tokens ('tokens'), the head entity information ('subj'), and tail entity information ('obj')
```python
from spanbert import SpanBERT
bert = SpanBERT(pretrained_dir="./pretrained_spanbert")
examples = [
        {'tokens': ['Bill', 'Gates', 'is', 'the', 'founder', 'of', 'Microsoft'], 'subj': ('Bill Gates', 'PERSON', (0,1)), "obj": ('Microsoft', 'ORGANIZATION', (6,6))},
        {'tokens': ['Bill', 'Gates', 'is', 'the', 'founder', 'of', 'Microsoft'], 'obj': ('Bill Gates', 'PERSON', (0,1)), "subj": ('Microsoft', 'ORGANIZATION', (6,6))}
        ]
preds = bert.predict(examples)
print(preds)
```


## Contact
If you have any questions, please contact Giannis Karamanolakis `<gkaraman@cs.columbia.edu>`.
