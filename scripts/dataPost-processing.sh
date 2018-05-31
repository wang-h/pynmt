#!/bin/bash
# author: Hao WANG
echo "Usage: bash dataPost-processing.sh WORKSPACE OUTPUT_FILE LANG_F LANG_E"
LANG_F=$3
LANG_E=$4
TASK=${LANG_F}-${LANG_E}

MOSES_SCRIPT=/itigo/files/Tools/accessByOldOrganization/TranslationEngines/mosesdecoder-RELEASE-2.1.1/scripts
#Evaluation
EVAL_SCRIPTS=/itigo/files/Tools/accessByOldOrganization/MTEvaluation

WORKSPACE=$1
OUTPUT=$2

if [ ${LANG_F}  == 'en' ]; then
  CORPUS=/itigo/Uploads/WMT2018/${LANG_F}-${LANG_E}/orig_clean
else
  CORPUS=/itigo/Uploads/WMT2018/${LANG_E}-${LANG_F}/orig_clean
fi

mkdir -p ${WORKSPACE}/eval

cp $CORPUS/test.${LANG_E} ${WORKSPACE}/eval/ref.${LANG_E}
cp ${WORKSPACE}/${OUTPUT} ${WORKSPACE}/eval/pred.${LANG_E}


cd ${WORKSPACE}/eval

for file in pred; do
    cat ${file}.${LANG_E} |\
    sed -r 's/(@@ )|(@@ ?$)//g'\
    > detok 
done

# For BLEU
perl ${MOSES_SCRIPT}/generic/multi-bleu.perl ref.${LANG_E} < detok 


