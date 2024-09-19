export CUDA_VISIBLE_DEVICES="3"
export WANDB_API_KEY=d3be0188f8e6de441fe26438708884794c8db33f
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/chengqi/miniconda3/envs/dev_co/bin/python train_cotasp.py --seed 330 --wandb_project_name final_results --default_beta 0.3 --is_store_everything