module load cuda

cd ShortcutGrammar/

cp ./tsv_data/inp_data/$1_$2.tsv ./ShortcutGrammar/data/$1/test.tsv

python conv_label.py -d $1 -s test

sh eval.sh $1 $4

python sst2_features_final.py -d $1 -s $2 -k 3

cd ../

python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_concept_constraint -nr 1

python lexical_constraints_exemplars_without_pos_and_concepts.py \
        -i ./tsv_data/inp_data/$1_$2.tsv \
        -p ./tsv_data/out_data \
        -j ./generation_data/$1_$2_concept_constraint/generated_predictions.jsonl \
        -ds $1 \
        -o $1_$2_final_constraint\
        -d $3

python add_json_meta_data.py -d $1 -s $2
echo "Added metadata for: $1_$2"

python tsv_to_jsonl.py -d $1 -s $2
echo "Solo json created for: $1_$2"

cp ./tsv_data/out_data/$1/$1_$2_solo_constraint.json ./generation_data/
cp ./tsv_data/out_data/$1/$1_$2_final_constraint_abs.json ./generation_data/

python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_final_constraint_abs -nr 1

python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_solo_constraint -nr 3

python abs_format.py -d $1 -s $2

echo "$1"
python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_abs_prompt_1 -nr 1

echo "$2"
python generate_data_pipe.py -m meta-llama/Llama-2-13b-chat-hf -c $1_$2_abs_prompt_2 -nr 1

echo "Creating final aug file."
python json_to_tsv.py -d $1 -s $2