# SpanBERT for Relation Extraction from Web Documents
This repository is a fork from [SpanBERT](https://github.com/facebookresearch/SpanBERT) by Facebook Research, which contains code and models for the paper: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).

We have adapted the SpanBERT scripts to support relation extraction from general documents beyond the TACRED dataset. 

## Install Requirements

```bash
pip install -r requirements.txt
```

## Download Pre-Trained SpanBERT (Fine-Tuned in TACRED)
SpanBERT has the same model configuration as [BERT](https://github.com/google-research/bert) but it differs in
both the masking scheme and the training objectives.

* Architecture: 24-layer, 1024-hidden, 16-heads, 340M parameters
* Fine-tuning Dataset: [TACRED](https://nlp.stanford.edu/projects/tacred/) ([42 relation types](https://github.com/gkaramanolakis/SpanBERT/blob/master/relations.txt))

To download the fine-tuned SpanBERT model run: 

```bash
./download_finetuned.sh
```

## Apply SpanBERT 

```python
from spanbert import SpanBERT
bert = SpanBERT(pretrained_dir="./pretrained_spanbert")
```
Input is a list of dicts, where each dict contains the sentence tokens ('tokens'), the subject entity information ('subj'), and object entity information ('obj'). Entity information is provided as a tuple: (\<Entity Name\>, \<Entity Type\>, (\<Start Location\>, \<End Location\>))

```python
examples = [
        {'tokens': ['Bill', 'Gates', 'stepped', 'down', 'as', 'chairman', 'of', 'Microsoft'], 'subj': ('Bill Gates', 'PERSON', (0,1)), "obj": ('Microsoft', 'ORGANIZATION', (7,7))},
        {'tokens': ['Bill', 'Gates', 'stepped', 'down', 'as', 'chairman', 'of', 'Microsoft'], 'subj': ('Microsoft', 'ORGANIZATION', (7,7)), 'obj': ('Bill Gates', 'PERSON', (0,1))},
        {'tokens': ['Zuckerberg', 'began', 'classes', 'at', 'Harvard', 'in', '2002'], 'subj': ('Zuckerberg', 'PERSON', (0,0)), 'obj': ('Harvard', 'ORGANIZATION', (4,4))}
        ]
preds = bert.predict(examples)
```

Output is a list of the same length as the input list, which contains the SpanBERT predictions and confidence scores

```python
print("Output: ", preds)
# Output: [('per:employee_of', 0.99), ('org:top_members/employees', 0.98), ('per:schools_attended', 0.98)]
```

## Contact
If you have any questions, please contact Giannis Karamanolakis `<gkaraman@cs.columbia.edu>`.
