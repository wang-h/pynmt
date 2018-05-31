#!/bin/bash
# author: Hao WANG
echo "Usage: bash dataPost-processing.sh WORKSPACE OUTPUT_FILE LANG_F LANG_E"
LANG_F=$2
LANG_E=$3
TASK=${LANG_F}-${LANG_E}

MOSES_SCRIPT=/itigo/files/Tools/accessByOldOrganization/TranslationEngines/mosesdecoder-RELEASE-2.1.1/scripts

CORPUS=$1


truecase_model=/itigo/Uploads/WMT2018/${LANG_F}-${LANG_E}/true


for prefix in newstest2018; do
  perl  ${MOSES_SCRIPT}/ems/support/input-from-sgm.perl \
    < ${CORPUS}/$prefix-${LANG_F}${LANG_E}-src-ts.${LANG_F}.sgm \
    | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_F} \
    | ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -a -l ${LANG_F} \
    | ${MOSES_SCRIPT}/recaser/truecase.perl   -model ${truecase_model}/truecase-model.${LANG_F} \
    > ${CORPUS}/../${LANG_F}-${LANG_E}/newstest2018.${LANG_F}

  perl  ${MOSES_SCRIPT}/ems/support/input-from-sgm.perl \
    < ${CORPUS}/$prefix-${LANG_E}${LANG_F}-src-ts.${LANG_E}.sgm \
    | ${MOSES_SCRIPT}/tokenizer/normalize-punctuation.perl -l ${LANG_E} \
    | ${MOSES_SCRIPT}/tokenizer/tokenizer.perl -a -l ${LANG_E} \
    | ${MOSES_SCRIPT}/recaser/truecase.perl   -model ${truecase_model}/truecase-model.${LANG_E} \
    > ${CORPUS}/../${LANG_F}-${LANG_E}/newstest2018.${LANG_E}

 done