# How to run iPhone UMI pipeline

## Setup

You should be able to run these commands directly using the data since it can be accessed by everyone.
I recommend symlinking `demonstration_processing/tmp_demonstrations` and `demonstration_processing/tmp_sessions` to folders `umi_data/iphone/demonstrations` and `umi_data/iphone/sessions`. The `umi_data` folder would exist in the same parent folder as the root of this repo (so `<ROOT>/umi_day/...` and `<ROOT>/umi_data/...`). Then you can take advantage of the VSCode workspace in `umi_day/.vscode/workspace.code-workspace`.

If running on server with no GUI you should install headless open3d by running `umi_day/scripts/install_open3d_headless.sh`.

## Group, time align and visualize iPhone and GoPro demonstrations
```bash
python process_demos.py group.iphone_dir=/store/real/public/austin_chuer_data/12_04_time_align/iphone group.right_gopro_dir=/store/real/public/austin_chuer_data/12_04_time_align/right_go_pro filters.session_name=austin-1204-utc
```

## Create an UMI session
You could also do something like `--input_session_filters austin-1204-.*` if you had multiple UMI day sessions you wanted to match. Note that `--output_session_name austin-1204` is the name of the output session to create
```bash
python create_session.py --input_session_filters austin-1204-utc --output_session_name austin-1204 
```

## Build an UMI dataset from the session
```bash
python build_umi_dataset.py tmp_sessions/austin-1204
# generates dataset_plan.pkl and dataset.zarr.zip
```

## Train an UMI policy
```bash
# clone UMI repo then and create `umi` conda environment, then...
cd universal_manipulation_interface
CUDA_VISIBLE_DEVICES="0,1,5" accelerate launch --num_processes 3 train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=/store/real/auspatel/umi_day_proj/umi_day/umi_day/demonstration_processing/tmp_sessions/austin-1204/dataset_austin-1204_2024-12-05_00-08-06.zarr.zip task_name="enter_your_task_name"
```
