### conda env
conda activate ascii



### teleop
python -m lerobot.teleoperate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"

python -m lerobot.teleoperate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}" --display_data=true



### record

python -m lerobot.record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/cheese_slicer_v4 --dataset.num_episodes=15 --dataset.episode_time_s 180 --dataset.reset_time_s 30 --dataset.single_task="Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_v4"  --dataset.push_to_hub=true --display_data=false 

- dataset path
C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_v4


--- 
    check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)
  File "C:\Users\meets\Projects\solobot\lerobot\src\lerobot\datasets\utils.py", line 574, in check_timestamps_sync
    raise ValueError(
ValueError: One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.

### upload dataset
huggingface-cli upload --repo-type dataset tinkerbuggy/cheese_slicer_v4 C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_v4


# training - done on vast.ai


### Creating Instance

Created a new instance rtx4090 using nvidia cuda template

https://docs.vast.ai/documentation/instances/sshscp#windows-putty-guide shows how to add ssh keys before connecting to these machines.

on windows:
ssh-keygen -t ed25519 -C "my-gmail@gmail.com"
- saved to C:\Users\meets/.ssh/id_ed25519)
cat C:\Users\meets/.ssh/id_ed25519.pub
copy and paste it to "add ssh key" section 

ssh -p 20800 root@213.181.123.57 -L 8080:localhost:8080

### install uv and create venv 

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
 uv venv --python 3.10
 source .venv/bin/activate

### install lerobot for ascii robot
git clone https://github.com/meetsitaram/lerobot
cd lerobot
git checkout ascii
- install missing python dev tools to avoid error "'/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1"
sudo apt-get install python3.10-dev
sudo apt-get install ffmpeg
uv pip install -e .

### login to hugging face and wandb
export HUGGINGFACE_TOKEN=my-token
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
wandb login
wandb access key my-accesskey


### download dataset
huggingface-cli download tinkerbuggy/cheese_slicer_v3 --repo-type dataset --local-dir /root/.cache/huggingface/lerobot/tinkerbuggy/cheese_slicer_v3


huggingface-cli download tinkerbuggy/cheese_slicer_v4 --repo-type dataset --local-dir /root/.cache/huggingface/lerobot/tinkerbuggy/cheese_slicer_v4


huggingface-cli download tinkerbuggy/cheese_slicer_v5 --repo-type dataset --local-dir /root/.cache/huggingface/lerobot/tinkerbuggy/cheese_slicer_v5


### run training


lerobot-train  --dataset.repo_id=tinkerbuggy/cheese_slicer_v3 --policy.type=act --output_dir=outputs/train/act_cheese_slicer_v3 --policy.repo_id=tinkerbuggy/cheese_slicer_v3  --policy.device=cuda  --wandb.enable=true


- NOTE***  (dont delete after training is done) - to cleanup directory:  rm -rf outputs/train/act_cheese_slicer_v3 (if duplicate)


- with dataset v4 - 16 episodes


lerobot-train  --dataset.repo_id=tinkerbuggy/cheese_slicer_v4 --policy.type=act --output_dir=outputs/train/act_cheese_slicer_v4 --policy.repo_id=tinkerbuggy/cheese_slicer_v4  --policy.device=cuda  --wandb.enable=true


- with dataset v4 - 16 episodes (removed episodes 16 and 17)


lerobot-train  --dataset.repo_id=tinkerbuggy/cheese_slicer_v5 --policy.type=act --output_dir=outputs/train/act_cheese_slicer_v5 --policy.repo_id=tinkerbuggy/cheese_slicer_v5  --policy.device=cuda  --wandb.enable=true


### final training command to run in background
nohup lerobot-train  --dataset.repo_id=tinkerbuggy/cheese_slicer_v5 --policy.type=act --output_dir=outputs/train/act_cheese_slicer_v5 --policy.repo_id=tinkerbuggy/cheese_slicer_v5  --policy.device=cuda  --wandb.enable=true & 


tail -f nohup.out


### upload model at 60K steps
 huggingface-cli upload --repo-type model tinkerbuggy/cheese_slicer_v5_60k outputs/train/act_cheese_slicer_v5/checkpoints/060000/pretrained_model

huggingface-cli upload --repo-type model tinkerbuggy/cheese_slicer_v5_80k outputs/train/act_cheese_slicer_v5/checkpoints/080000/pretrained_model

- model url
https://huggingface.co/tinkerbuggy/cheese_slicer_v5_60k
https://huggingface.co/tinkerbuggy/cheese_slicer_v5_80k



# Running with model

### download model
huggingface-cli download tinkerbuggy/cheese_slicer_v5_80k --repo-type model --local-dir C:\Users\meets\.cache\huggingface\hub\models--tinkerbuggy--cheese_slicer_v5_80k


### run the model

conda activate ascii

lerobot-record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/eval_act-cheese_slicer_v5 --dataset.num_episodes=1 --dataset.episode_time_s 180 --dataset.reset_time_s 30 --dataset.single_task="Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_v5"  --display_data=true --policy.path=tinkerbuggy/cheese_slicer_v5_80k

- problem with creating symlink (fixed by download it manually to the directory)
C:\Users\meets\miniconda3\envs\ascii\lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\meets\.cache\huggingface\hub\models--tinkerbuggy--cheese_slicer_v5_80k. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development


- problem with policy path
  File "C:\Users\meets\Projects\solobot\lerobot\src\lerobot\record.py", line 180, in __post_init__
    self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
  File "C:\Users\meets\Projects\solobot\lerobot\src\lerobot\configs\policies.py", line 207, in from_pretrained
    return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\meets\\AppData\\Local\\Temp\\tmpcu8zg55f'

added this extra check in policies.py before reading the config file
        print(config_file)
        if(config_file is None or not os.path.exists(config_file)):
            raise FileNotFoundError(f"Config file not found: {config_file}")

shows that the file is at C:\Users\meets\.cache\huggingface\hub\models--tinkerbuggy--cheese_slicer_v5_80k\snapshots\cce3f5264dec67cd15e49fb2275ba16e3949bd6d\config.json

- trying opening powershell and run as admin
conda activate ascii
cd C:\Users\meets\Projects\solobot\lerobot\

lerobot-record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/eval_act-cheese_slicer_v5 --dataset.num_episodes=1 --dataset.episode_time_s 180 --dataset.reset_time_s 30 --dataset.single_task="Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\act_cheese_slicer_v5"  --display_data=true --policy.path=tinkerbuggy/cheese_slicer_v5_80k

--  trying to open in read-only mode (didn't help) (when running from vscode, will try powershell)
        with open(config_file, 'r') as f:
            config = json.load(f)

- error is at:
 File "C:\Users\meets\Projects\solobot\lerobot\src\lerobot\configs\policies.py", line 212, in from_pretrained
    return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)

    (didn't fix)

- made this change for creating tmp files in policies.py
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "act_config.json")
            with open(temp_file_path, 'w+') as f:
        # with tempfile.NamedTemporaryFile("w+") as f:

### go with replay
python -m lerobot.replay --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --dataset.repo_id=tinkerbuggy/cheese_slicer_v5 --dataset.episode=9

lerobot-replay --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"  --dataset.repo_id=tinkerbuggy/cheese_slicer_v4 --dataset.episode=10  --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_v4" 



### recording a new dataset
python -m lerobot.record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/cheese_slicer_fine --dataset.num_episodes=15 --dataset.episode_time_s 180 --dataset.reset_time_s 30 --dataset.single_task="Fine Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_fine"  --dataset.push_to_hub=true --display_data=false 


- dataset path
C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_fine

### upload new dataset
huggingface-cli upload --repo-type dataset tinkerbuggy/cheese_slicer_fine C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_fine


### replay with new fine-cheese-slice episodes
- good episodes: 2, 5, 9

lerobot-replay --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"  --dataset.repo_id=tinkerbuggy/cheese_slicer_fine --dataset.episode=2  --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\cheese_slicer_fine" 

### download 40k model
huggingface-cli download tinkerbuggy/cheese_slicer_fine_h100_40k --repo-type model --local-dir C:\Users\meets\.cache\huggingface\hub\models--tinkerbuggy--cheese_slicer_fine_h100_40k

### run 40k model
lerobot-record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"  --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/eval_act_cheese_slicer_fine_h100_40k --dataset.num_episodes=1 --dataset.episode_time_s 300 --dataset.reset_time_s 30 --dataset.single_task="Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\act_cheese_slicer_fine_h100_40k"  --display_data=true --policy.path=tinkerbuggy/cheese_slicer_fine_h100_40k

- eval dataset path
C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\act_cheese_slicer_fine_h100_40k


### teleop
python -m lerobot.teleoperate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"


### run final model
lerobot-record --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"  --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}"  --dataset.repo_id=tinkerbuggy/eval_act_cheese_slicer_fine_h100 --dataset.num_episodes=1 --dataset.episode_time_s 300 --dataset.reset_time_s 30 --dataset.single_task="Slice Cheese" --dataset.root="C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\act_cheese_slicer_fine_h100"  --display_data=true --policy.path=tinkerbuggy/cheese_slicer_fine_h100

- eval dataset path
C:\Users\meets\Projects\solobot\lerobot\tinkerbuggy\act_cheese_slicer_fine_h100


huggingface-cli download tinkerbuggy/cheese_slicer_fine_h100_80k --repo-type model --local-dir C:\Users\meets\.cache\huggingface\hub\models--tinkerbuggy--cheese_slicer_fine_h100_80k