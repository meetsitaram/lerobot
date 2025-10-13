

###  assembly
- follow https://huggingface.co/docs/lerobot/so101

### environment setup
- followed https://huggingface.co/docs/lerobot/installation
- had trouble running `pip install -e .` directly from anaconda prompt and got error:
```
    path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
  File "C:\Users\meets\anaconda3\envs\solobot\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 209, in get_win_folder_from_registry
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

- to address this problem, i started vscode from anaconda navigator and pip install works fine from the vscode terminal opened from here.

### pip install failing
`pip install 'lerobot[all]'` fails with errors:

### ports
COM7 - leader
COM5 - follower

### setup motor ids
lerobot-setup-motors  --robot.type=so101_follower --robot.port=COM5
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=COM5

###
uv - python dependency management - check it out

### notes
Get-ExecutionPolicy
Set-ExecutionPolicy RemoteSigned

### solo install
cd solo-server
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
uv pip install solo-server

<!-- uv pip install lerobot -->
<!-- uv pip install 'lerobot[feetech]' -->

## conda issues
i have to add conda `C:\Users\meets\anaconda3\Scripts\` to `PATH`system environment variables to get it working from vscode terminal
- reinstalled miniconda to C:\Users\meets\miniconda3\Scripts\conda.exe
- used copilot to set miniconda as default terminal

these two commands load on opening a new terminal in vscode
```sh
C:/Users/meets/miniconda3/Scripts/activate
conda activate base
```

conda create -y -n ascii python=3.10

conda info --envs
conda activate ascii
pip install -e .
pip install -e ".[feetech]"


conda create -y -n solo python=3.10
conda install ffmpeg -c conda-forge
pip install solo-server


### run
.venv/Scripts/activate
solo robo --type lerobot --teleop
solo robo --type lerobot --calibrate follower
solo robo --type lerobot --calibrate leader

### lerobot commands
lerobot-calibrate --robot.type=so101_follower --robot.port=COM9 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM8  --teleop.id=my_awesome_leader_arm

python -m lerobot.teleoperate --robot.type=so101_follower --robot.port=COM9 --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=COM8 --teleop.id=my_awesome_leader_arm


### config
- saving config to C:\Users\meets/.solo_server\config.json instead of local repo folder

### opencv issue
`pip install opencv` fixed issue connecting to cameras

### so101-koch
conda activate solo

pip install -e .
pip install ".[feetech]"
pip install ".[dynamixel]"

lerobot-find-port
lerobot-setup-motors --teleop.type=ascii_leader --teleop.port=COM8
lerobot-setup-motors --robot.type=ascii_follower --robot.port=COM11

lerobot-calibrate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"

lerobot-calibrate --teleop.type=ascii_leader --teleop.port=COM8  --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"

python -m lerobot.teleoperate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot"

python -m lerobot.teleoperate --robot.type=ascii_follower --robot.port=COM11 --robot.id=ascii_follower --robot.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --teleop.type=ascii_leader --teleop.port=COM8 --teleop.id=ascii_leader --teleop.calibration_dir="C:\Users\meets\Projects\solobot\lerobot" --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" --display_data=true


# recording
Press Right Arrow (→): Early stop the current episode or reset time and move to the next.
Press Left Arrow (←): Cancel the current episode and re-record it.
Press Escape (ESC): Immediately stop the session, encode videos, and upload the dataset.


### smolvla


pip install -e ".[feetech,smolvla]"

  File "C:\Users\meets\Projects\solobot\lerobot-upstream\lerobot\src\lerobot\processor\pipeline.py", line 665, in _load_config
    raise FileNotFoundError(
FileNotFoundError: Could not find 'policy_preprocessor.json' on the HuggingFace Hub at 'lerobot/smolvla_base'
