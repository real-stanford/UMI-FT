# umi_day

## Setup

Clone this repo and clone submodules
```bash
git clone --recurse-submodules git@github.com:austinapatel/umi_day.git
```

### Setup mamba env:
```bash
# this automatically does editable install of `umi_day` package from local source
mamba env create -f environment.yml
mamba activate umi_day
```

### Headless setup if you don't have display
```bash
cd scripts && . install_open3d_headless.sh
```
