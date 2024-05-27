python src/run.py  \
   --eval_on $1 \
   --test_run $2\
   --load_from ./output/$1/pcfg/ \
   --model_type pcfg \
   --preterminals 64 \
   --nonterminals 32 \
   --eval_batch_size 1 \
   --output_dir output/$1/ \
   --max_length 128 \
   --max_eval_length 128 \
   --cache "";

   # --load_from output/$1/pcfg/ \
