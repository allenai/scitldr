#!/bin/bash
python to_stories.py $TASK
python make_datafiles.py --stories_dir $TASK/stories --urldir $TASK/mapping --finished_files_dir $TASK

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
for SPLIT in train val
do
  for LANG in source target
  do
    python -m multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

# Make dataset with crl tags
python build_ctrl_datasets.py $TASK

python to_stories.py $TASK/ctrl
python make_datafiles.py --stories_dir $TASK/ctrl/stories --urldir $TASK/ctrl/mapping --finished_files_dir $TASK/ctrl

for SPLIT in train val
do
  for LANG in source target
  do
    python -m multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/ctrl/$SPLIT.$LANG" \
    --outputs "$TASK/ctrl/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/ctrl/train.bpe" \
  --validpref "${TASK}/ctrl/val.bpe" \
  --destdir "${TASK}/ctrl-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;