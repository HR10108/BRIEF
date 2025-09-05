# #!/bin/bash
train_file=$1 
model=$2 # path to model
output_path=$3 # path to output
dims=$4 # dimension of projection, can be a list
gradient_type=$5
INFO_TYPE=$6
max_length=$7


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m brief.get_grads.get_info \
--train_file $train_file \
--info_type $INFO_TYPE \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type \
--max_length $max_length