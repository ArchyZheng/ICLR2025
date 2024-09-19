export CUDA_VISIBLE_DEVICES="2"
export WANDB_API_KEY=d3be0188f8e6de441fe26438708884794c8db33f
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# /home/chengqi/miniconda3/envs/dev_co/bin/python train_cotasp.py --seed 110 --wandb_project_name sep_adaptive_beta --adaptive_beta True --beta_lambda 0.4
/home/chengqi/miniconda3/envs/dev_co/bin/python train_cotasp.py --seed 110 --wandb_project_name sep_adaptive_beta_and_input_sensitive --adaptive_beta True --beta_lambda 0.4 --use_input_sensitive True
# /home/chengqi/miniconda3/envs/dev_co/bin/python train_cotasp.py --seed 110 --wandb_project_name multi_head --adaptive_beta True
# /home/chengqi/miniconda3/envs/dev_co/bin/python train_cotasp_only_task_5.py --seed 330