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
Please follow the installation instructions [here](https://github.com/pltrdy/files2rouge).

### Model Weights
[`catts.tldr-ao`](https://storage.cloud.google.com/skiff-models/scitldr/catts.tldr-ao.pt)

[`catts.tldr-aic`](https://storage.cloud.google.com/skiff-models/scitldr/catts.tldr-aic.pt)

[`catts-xsum.tldr-ao`](https://storage.cloud.google.com/skiff-models/scitldr/catts-xsum.tldr-ao.pt)

[`catts-xsum.tldr-aic`](https://storage.cloud.google.com/skiff-models/scitldr/catts-xsum.tldr-aic.pt)

[`bart.tldr-ao`](https://storage.cloud.google.com/skiff-models/scitldr/bart.tldr-ao.pt)

[`bart.tldr-aic`](https://storage.cloud.google.com/skiff-models/scitldr/bart.tldr-aic.pt)

[`bart-xsum.tldr-ao`](https://storage.cloud.google.com/skiff-models/scitldr/bart-xsum.tldr-ao.pt)

[`bart-xsum.tldr-aic`](https://storage.cloud.google.com/skiff-models/scitldr/bart-xsum.tldr-aic.pt)


### Data Preprocessing
In order to format the data to work for the Fairseq library, run:
```bash
cd SciTLDR-Data
export TASK=SciTLDR-A # Choose from {A, AIC, FullText}
chmod +x make_datafiles.sh
./make_datafiles.sh # BPE preprocess
```
`$TASK/ctrl` contains the dataset formatted with the control codes.

### Generation
This code takes in a `test.source` file, in which each line is an input and outputs a `test.hypo` file with the predictions. See [decoder_params](decoder_params.md) for optimal decoder parameters for each version of the model.
```bash
python scripts/generate.py /path/to/modeldir/ SciTLDR-Data/SciTLDR-A/ctrl ./ --beam 2 --lenpen 0.4 --test_fname test.hypo
```

### Evaluation
This script is a wrapper around ROUGE that takes in a `test.hypo` file and compares to a `test.jsonl` file.
```bash
python scripts/cal-rouge.py /path/to/test.hypo SciTLDR-Data/SciTLDR-A/test.jsonl --workers 1
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
