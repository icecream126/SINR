####################################################################################################

## Create new session 
session=gauss_era5_fine
tmux new-session -d -s $session

## define "project_name" for wandb 
project_name=re_spatial

### Set window:1 ####
window=1
gpu=0
tmux new-window -t $session:$window
tmux send-keys "conda activate hm_wire" C-m

## target=11
## step_size=1e-3
## run_name=flag-${step_size}_target-${target} # define "run_name" for wandb 
run_name=gauss
dataset_dir=
## tmux send-keys -t $session:$window "python main.py --step_size ${step_size} --aug_type FLAG --epochs 1000 --dis 4.0 --target ${target} --max_com 1_2 --num_hidden 66 --gradient_clip 1.0 --gpu ${gpu} --wandb --project_name ${project_name} --group_name ${run_name} --run_name ${run_name}" C-m
tmux send-keys -t $session:$window "CUDA_VISIBLE_DEVICES=${gpu} python src/main_superres.py "


### Set window:2 ####
window=2
gpu=3
tmux new-window -t $session:$window
tmux send-keys "conda activate empsn" C-m

target=0
run_name=pointnet-atom-pe_-${target}
tmux send-keys -t $session:$window "python main.py --aug_type pointnet-atom --epochs 1000 --dis 4.0 --target ${target} --max_com 1_2 --num_hidden 66 --gradient_clip 1.0 --gpu ${gpu} --wandb --project_name ${project_name} --group_name ${run_name} --run_name ${run_name}" C-m