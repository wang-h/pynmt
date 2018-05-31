#!/bin/bash
# author: Hao WANG
echo "Usage: bash $0 prefix LANG_F [=en] LANG_E"

prefix=$1
LANG_F=$2
LANG_E=$3
TASK=${LANG_F}-${LANG_E}
spm_encode=spm_encode
spm_model=/WMT2018/$TASK/$TASK.spm.model
truecase_model=/path-to/$TASK/true
MOSES_SCRIPT=/path-to/mosesdecoder-RELEASE-2.1.1/scripts


mkdir -p SPM

cat $prefix.${LANG_F} \
            | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_F}  \
            | ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -a -l ${LANG_F}  \
            | ${MOSES_SCRIPT}/recaser/truecase.perl   -model ${truecase_model}/truecase-model.${LANG_F}  \
            | $spm_encode --model=$spm_model --output_format=piece \
            > SPM/synthetic.${LANG_E}-${LANG_F}.${LANG_F}
cat $prefix.${LANG_E} \
            | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_E}  \
            | ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -a -l ${LANG_E}  \
            | ${MOSES_SCRIPT}/recaser/truecase.perl   -model ${truecase_model}/truecase-model.${LANG_E}  \
            | $spm_encode --model=$spm_model --output_format=piece \
            > SPM/synthetic.${LANG_F}-${LANG_E}.${LANG_E}

