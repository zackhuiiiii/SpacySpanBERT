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

```bash
./download_finetuned.sh
```

### TACRED

```bash
python code/run_tacred.py 
```


## Contact
If you have any questions, please contact Giannis Karamanolakis `<gkaraman@cs.columbia.edu>` or create a Github issue.
