export CUDA_VISIBLE_DEVICES="0"
export WANDB_API_KEY=d3be0188f8e6de441fe26438708884794c8db33f
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/chengqi/.conda/envs/dev_co/bin/python train_cotasp.py \
    --seed 110 \
    --wandb_project_name rebuttal_results \
    --use_input_sensitive \
    --default_beta 0.3 \
    --is_store_everything \
    --reset_log_std \
    --dormant_type sensitivity \
    --layer_neuron_threshold 0.6 \
    --env_name salina/halfcheetah/forgetting \
    --eval_interval 200000 \
    --calculate_layer_sensitivity_interval 80000 \