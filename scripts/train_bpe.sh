#!/bin/bash
# author: Hao WANG

BPE_SCRIPTS=/path-to/BPE
CORPUS=/path-to/WMT2018/en-tr/orig_clean
MONO_CORPUS=/path-to/WMT2018/en-tr/mono_data
mkdir -p BPE
${BPE_SCRIPTS}/learn_joint_bpe_and_vocab.py --input $CORPUS/train.$1 $CORPUS/train.$2 --min-frequency 5 --symbols 30000 --output BPE/vocab.$1-$2 --write-vocabulary BPE/vocab.$1 BPE/vocab.$2 
${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/train.$1 --codes BPE/vocab.$1-$2 --output BPE/train.$1
${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/dev.$1 --codes BPE/vocab.$1-$2 --output BPE/dev.$1
${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/test.$1 --codes BPE/vocab.$1-$2 --output BPE/test.$1

${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/train.$2 --codes BPE/vocab.$1-$2 --output BPE/train.$2
${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/dev.$2 --codes BPE/vocab.$1-$2 --output BPE/dev.$2
${BPE_SCRIPTS}/apply_bpe.py --input $CORPUS/test.$2 --codes BPE/vocab.$1-$2 --output BPE/test.$2

# ${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.orig.$1 --codes BPE/vocab.$1-$2 --output BPE/synthetic.orig.$1
# ${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.orig.$2 --codes BPE/vocab.$1-$2 --output BPE/synthetic.orig.$2

#${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.orig.2M.$1 --codes BPE/vocab.$1-$2 --output BPE/synthetic.orig.2M.$1
#${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.orig.2M.$2 --codes BPE/vocab.$1-$2 --output BPE/synthetic.orig.2M.$2


#${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.trans.$1 --codes BPE/vocab.$1-$2 --output BPE/synthetic.trans.$1-$2.$1
#cp BPE/synthetic.orig.$2 BPE/synthetic.trans.$1-$2.$2
#${BPE_SCRIPTS}/apply_bpe.py --input $MONO_CORPUS/synthetic.orig.$2 --codes BPE/vocab.$1-$2 --output BPE/synthetic.orig.$2

