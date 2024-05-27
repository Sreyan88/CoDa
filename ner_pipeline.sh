pos=$4

if [ "$pos" = "0" ]; then
    echo "Executing without pos"
    python ./Lexical-Constraints/lexical_constraints_ner_without_pos.py \
            -i ./tsv_data/inp_data/$1_$2.tsv \
            -p ./tsv_data/out_data \
            -ds $1 \
            -o $1_$2_final_constraint\
            -d $3
else
    echo "Executing with pos"
    python ./Lexical-Constraints/lexical_constraints_ner_pos.py \
            -i ./tsv_data/inp_data/$1_$2.tsv \
            -p ./tsv_data/out_data \
            -ds $1 \
            -o $1_$2_final_constraint\
            -d $3
fi

python add_json_meta_data.py -d $1 -s $2
echo "Added metadata for: $1_$2"

python tsv_to_jsonl_ner.py -d $1 -s $2
echo "Solo json created for: $1_$2"

cp ./tsv_data/out_data/$1/$1_$2_solo_constraint.json ./generation_data/
cp ./tsv_data/out_data/$1/$1_$2_final_constraint_abs.json ./generation_data/

python abs_format_ner.py -d $1 -s $2

python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_solo_constraint -nr 5

python json_to_conll.py -d $1 -s $2