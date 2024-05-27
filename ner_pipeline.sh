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

