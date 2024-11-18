export CUDA_VISIBLE_DEVICES="0"
export WANDB_API_KEY=d3be0188f8e6de441fe26438708884794c8db33f
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/chengqi/.conda/envs/dev_co/bin/python train_cotasp.py \
    --seed 220 \
    --wandb_project_name rebuttal_results \
    --default_beta 0.3 \
    --is_store_everything \
    --reset_log_std \
    --dormant_type sensitivity \
    --layer_neuron_threshold 0.5 \
    --env_name salina/halfcheetah/robustness \
    --eval_interval 2000 \
    --use_input_sensitive \
    --calculate_layer_sensitivity_interval 3000 \


        # --seed 110",
        # --wandb_project_name", "rebuttal_results",
        # --env_name", "salina/halfcheetah/forgetting",
        # --default_beta", "0.3",
        # --is_store_everything",
        # --use_input_sensitive",
        # --reset_log_std",
        # --dormant_type", "sensitivity",
        # --eval_interval", "20000",
        # --calculate_layer_sensitivity_interval", "1000",