#!/bin/bash
# author: Hao WANG
echo "Usage: bash eval.sh OUTPUT_FILE LANG_F LANG_E"

OUTPUT=$1
LANG_F=$2
LANG_E=$3
TASK=${LANG_F}-${LANG_E}

MOSES_SCRIPT=/path-to/mosesdecoder-RELEASE-2.1.1/scripts
wmt18=/path-to/WMT2018/test2018

spm_decode=spm_decode

if [[ ${LANG_F} != 'en' ]]; then
  spm_model=/path-to/WMT2018/${LANG_E}-${LANG_F}/${LANG_E}-${LANG_F}.spm.model
else
  spm_model=/path-to/WMT2018/${LANG_F}-${LANG_E}/${LANG_F}-${LANG_E}.spm.model
fi


mteavl=${MOSES_SCRIPT}/generic/mteval-v13a.pl


src=${wmt18}/newstest2018-${LANG_F}${LANG_E}-src-ts.${LANG_F}.sgm

mkdir -p eval
cat $OUTPUT \
  | $spm_decode --model $spm_model --input_format=piece \
  | ${MOSES_SCRIPT}/recaser/detruecase.perl\
  | ${MOSES_SCRIPT}/tokenizer/detokenizer.perl -l ${LANG_E}\
  | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_E}\
  | ${MOSES_SCRIPT}/ems/support/wrap-xml.perl ${LANG_E} $src WAU\
   >eval/decoder-output.newstest2018.sgm



