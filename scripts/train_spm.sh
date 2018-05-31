src=$1
trg=$2
spm_train=spm_train
spm_decode=spm_decode
spm_encode=spm_encode

mkdir -p SPM
$spm_train --add_dummy_prefix False --input corpus --vocab_size=16000 --model_prefix $src-$trg.spm

for lang in $1 $2; do
    $spm_encode --model=$src-$trg.spm.model --output_format=piece < train.$lang > SPM/train.$lang
    $spm_encode --model=$src-$trg.spm.model --output_format=piece < dev2018.$lang > SPM/dev2018.$lang
    $spm_encode --model=$src-$trg.spm.model --output_format=piece < newstest2018.$lang > SPM/newstest2018.$lang
    
done