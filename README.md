# SciTLDR

This repository contains the dataset, model weights, and generation code for our paper "[TLDR: Extreme Summarization of Scientific Documents](https://arxiv.org/abs/2004.15011)". 

## Demo
A running demo of our model can be found [here](https://scitldr.apps.allenai.org).

## Dataset
SciTLDR is split in to a 60/20/20 train/dev/test split. For each file, each line is a json, formatted as follows

```
{
   "source":[
      "sent0",
      "sent1",
      "sent2",
      ...
   ],
   "source_labels":[binary list in which 1 is the oracle sentence],
   "rouge_scores":[precomputed rouge-1 scores],
   "paper_id":"PAPER-ID",
   "target":[
     "author-tldr",
      "pr-tldr0", 
      "pr-tldr1",
      ... 
   ],
   "title":"TITLE"
}
```
The keys `rouge_scores` and `source_labels` are not necessary for any code to run, but we provide precomputed Rouge scores to encourage future research. 

## Requirements
We use [Fairseq](https://fairseq.readthedocs.io) to train and evaluate our models. 
Install Fairseq as follows:
```bash
git clone fairseq repo #TODO figure out how to use specific version
cd fairseq
pip install --editable .
```

To install all other requirements, run `pip install -r requirements.txt`

For the evaluation, you will need `files2rouge`. 
Please install [my fork](https://github.com/isabelcachola/files2rouge) of the repo.

### Model Weights
[`bart.large.xsum.multitask-A`](https://storage.cloud.google.com/skiff-models/scitldr/ao_model.pt)

[`bart.large.xsum.multitask-AIC`](https://storage.cloud.google.com/skiff-models/scitldr/aic_model.pt)

### Data Preprocessing
In order to format the data to work for the Fairseq library, run:
```bash
cd SciTLDR-Data
export TASK=SciTLDR-A # Choose from {A, AIC, FullText}
chmod +x make_datafiles.sh
./make_datafiles.sh # BPE preprocess
```
`$TASK/ctrl` contains the dataset formatted with the control codes.

### Evaluation
This code takes in a `test.source` file, in which each line is an input and outputs a `test.hypo` file with the predictions. It imports a `test.jsonl` file as a reference and stores the rouge score in `test.hypo.score`.
```bash
python evaluate.py SciTLDR-Data/SciTLDR-A/ctrl /path/to/model/dir/ --checkpoint_file scitldr_ao_model.pt --beam 4 --lenpen 0.2
```
OR
```bas
python evaluate.py SciTLDR-Data/SciTLDR-AIC/ctrl /path/to/model/dir/ --checkpoint_file scitldr_aic_model.pt --beam 2 --lenpen 0.2 
```

### Citing
If you use our code, dataset, or model weights in your research, please cite "TLDR: Extreme Summarization of Scientific Documents."


```
@article{cachola2020tldr,
  title={{TLDR}: Extreme Summarization of Scientific Documents},
  author={Isabel Cachola and Kyle Lo and Arman Cohan and Daniel S. Weld},
  journal={arXiv:2004.15011},
  year={2020},
}
```

SciTLDR is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2). AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
