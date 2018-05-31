#!/bin/bash
# author: Hao WANG
echo "Usage: bash eval.sh OUTPUT_FILE LANG_F LANG_E"

OUTPUT=$1
LANG_F=$2
LANG_E=$3
TASK=${LANG_F}-${LANG_E}

MOSES_SCRIPT=/itigo/files/Tools/accessByOldOrganization/TranslationEngines/mosesdecoder-RELEASE-2.1.1/scripts
wmt17=/itigo/Uploads/WMT2018/wmt17-submitted-data
spm_decode=spm_decode

if [[ ${LANG_F} != 'en' ]]; then
  spm_model=/itigo/Uploads/WMT2018/${LANG_E}-${LANG_F}/${LANG_E}-${LANG_F}.spm.model
else
  spm_model=/itigo/Uploads/WMT2018/${LANG_F}-${LANG_E}/${LANG_F}-${LANG_E}.spm.model
fi
mteavl=${MOSES_SCRIPT}/generic/mteval-v13a.pl
src=${wmt17}/sgm/sources/newstest2017-${LANG_F}${LANG_E}-src.${LANG_F}.sgm
ref=${wmt17}/sgm/references/newstest2017-${LANG_F}${LANG_E}-ref.${LANG_E}.sgm 


mkdir -p eval

# cat $OUTPUT \
# > eval/decoder-output.sgm

  # | sed 's/\@\@ //g' \
  # | ${MOSES_SCRIPT}/recaser/detruecase.perl\
  # | ${MOSES_SCRIPT}/tokenizer/detokenizer.perl -l ${LANG_E}\
  # | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_E}\
  # | ${MOSES_SCRIPT}/ems/support/wrap-xml.perl ${LANG_E} $src WAU\


cat $OUTPUT \
  | $spm_decode --model $spm_model --input_format=piece \
  | ${MOSES_SCRIPT}/recaser/detruecase.perl\
  | ${MOSES_SCRIPT}/tokenizer/detokenizer.perl -l ${LANG_E}\
  | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_E}\
  | ${MOSES_SCRIPT}/ems/support/wrap-xml.perl ${LANG_E} $src WAU\
> eval/decoder-output.newstest2017.sgm
# For BLEU
perl $mteavl -s $src -r $ref -t  eval/decoder-output.newstest2017.sgm


