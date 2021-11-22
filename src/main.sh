#for  model_type in staktc staktns staktsk stakt;
for  model_type in staktnt;
do
  {
        date=$(date "+%Y%m%d%H%M%S")
        python -u main_2009.py  --dataset assist2009_updated --m 2 --n 2 --kernel_size 2  --model_type ${model_type} >> ${model_type}_${date}_m2n2k2_lr1e-3.log 2>&1 &&
        python -u main_2009.py  --dataset assist2009_updated --m 1 --n 1 --kernel_size 2  --model_type ${model_type} >> ${model_type}_${date}_m1n1k2_lr1e-3.log 2>&1 &&
        wait
  }
done
