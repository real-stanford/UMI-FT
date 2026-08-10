import hydra
from omegaconf import DictConfig
from utils.color_util import blue
from utils.generic_util import iterate_demonstrations
from process_stages.group import group_iphone_data
from process_stages.detect import detect_ar_tag_iphone
from process_stages.calibrate import calibrate_gripper_range_iphone
from process_stages.visualize import visualize_iphone_data
from process_stages.label import auto_label

@hydra.main(version_base="1.2", config_name="process_demos_iphone", config_path="./config")
def main(cfg: DictConfig):
    # Determine which stages to run
    all_stages = ['group', 'detect', 'calibrate', 'visualize', 'label']

    if cfg.stages is None:
        cfg.stages = all_stages

    stages = [stage for stage in cfg.stages if stage not in cfg.skip_stages]
    assert all([stage in all_stages for stage in stages])

    # Define demonstration iterator
    get_demonstration_iterator = lambda demo_type=None: iterate_demonstrations(cfg.demonstrations_dir, cfg.filters, demo_type)

    # Run requested stages
    if 'group' in stages:
        print(blue("--- GROUPING STAGE ---"))
        group_iphone_data(cfg.group)

    if 'detect' in stages:
        print(blue("\n--- DETECT AR TAG STAGE ---"))
        detect_ar_tag_iphone(get_demonstration_iterator, cfg.detect)

    if 'calibrate' in stages:
        print(blue("\n--- CALIBRATE GRIPPER STAGE ---"))
        calibrate_gripper_range_iphone(get_demonstration_iterator, cfg.calibrate)

    if 'label' in stages:
        print(blue("\n--- LABEL STAGE ---"))
        auto_label(get_demonstration_iterator, cfg.label)

    if 'visualize' in stages:
        print(blue("\n--- VISUALIZE STAGE ---"))
        visualize_iphone_data(get_demonstration_iterator, cfg.visualize) # add show in gui flag (real time render, not save to file), need to output file path to zarr for policy inference

if __name__ == '__main__':
    main()
