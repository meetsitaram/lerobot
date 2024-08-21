<!-- wsl -->
<!-- cd /mnt/c/Users/meets/Projects/lerobot -->

--- issues running in wsl or in docker
--- works fine in windows
--- data files downloaded to C:\Users\meets\.cache\huggingface\datasets

conda create -y -n aloha python=3.10
conda activate aloha
conda deactivate

pip install .
pip install ".[aloha, pusht]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  

<!-- pip install parquet-tools -->

<!-- python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0 -->

python lerobot/scripts/visualize_dataset.py --repo-id lerobot/aloha_sim_transfer_cube_human --episode-index 0
python lerobot/scripts/visualize_dataset.py --repo-id lerobot/unitreeh1_fold_clothes --episode-index 0

<!-- python lerobot/scripts/visualize_dataset.py --repo-id lerobot/aloha_sim_transfer_cube_human --episode-index 49 -->

python lerobot/scripts/train.py policy=act env=aloha env.task=AlohaInsertion-v0 dataset_repo_id=lerobot/aloha_sim_insertion_human 

<!-- python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10 eval.batch_size=10 -->

python lerobot/scripts/eval.py -p outputs/train/2024-06-14/18-58-49_aloha_act_default/checkpoints/002000/pretrained_model  eval.n_episodes=10




--------

 sudo apt-get install cmake build-essential

 make 

make build-gpu
make DEVICE=cuda test-act-ete-train
<!-- - mujovo physics engine -->


/etc/wsl.conf
[network]
generateResolvConf = false

The wayland library could not be loaded
 sudo apt-get install libwayland-dev

 xkbcommon_dl] Failed loading `libxkbcommon.so.0`.
 sudo apt-get install libxkbcommon-dev

 'main' panicked at 'called `Option::unwrap()` on a `None` value'
  10: wgpu::Instance::create_surface_unsafe


ImportError: libGL.so.1: cannot open shared object file: No such file or directory
sudo apt-get install libgl-dev  

docker issues with running in linux:
docker build -t lerobot:latest -f docker/lerobot-gpu/Dockerfile .
docker run -it lerobot

 neither WAYLAND_DISPLAY nor DISPLAY is set. 


checkpoint failure:
 OSError: [WinError 1314] A required privilege is not held by the client: 'C:\\Users\\meets\\Projects\\lerobot\\outputs\\train\\2024-06-14\\18-58-49_aloha_act_default\\checkpoints\\002000' -> 'outputs\\train\\2024-06-14\\18-58-49_aloha_act_default\\checkpoints\\last'

-----
robot-joint-space
cls - classification tokens
BERT - Bidirectional Encoder Representations from Transformers
lr - learning rate - change to speed up training 

------
vscode - preferences - settings - extensions - python - conda path
"C:\Users\meets\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\Anaconda Powershell Prompt.lnk"

settings - extensions - python interpreter path
import os,sys
os.path.dirname(sys.executable) -> os.path.dirname(sys.executable)
C:\Users\meets\anaconda3\envs\aloha\python.exe

configure tasks and add task:
        {
            "label": "visualize-aloha",
            "type": "shell",
            "command": "python lerobot/scripts/visualize_dataset.py --repo-id lerobot/aloha_sim_transfer_cube_human --episode-index 0"
        }

to debug, run train-aloha task and then python debugger - with launch configuration - select our running python process 


gazebo robot simulation tool
activate aloha environment, conda install conda-forge::gazebo and run gazebo.exe 

ros
xarco, urdf xml files
Dynamixel servo motors

fusion2urdf 
https://github.com/syuntoku14/fusion2urdf
copy fusion2urdf to C:\Users\meets\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\Scripts'
```sh
Copy-Item ".\URDF_Exporter\" -Destination "${env:APPDATA}\Autodesk\Autodesk Fusion 360\API\Scripts\" -Recurse 
Copy-Item ".\URDF_Exporter\" -Destination "${env:APPDATA}\Autodesk\Autodesk Fusion 360\API\Scripts\" -Recurse -force
```

ros-install
https://docs.ros.org/en/jazzy/Installation/Windows-Install-Binary.html


### rosrun to get urdf from xacro 
in ubuntu
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html
```
source /opt/ros/jazzy/setup.bash
sudo apt-get install ros-jazzy-xacro
ros2 run xacro xacro Basic_Robot.xacro --inorder > Basic_Robot.urdf
```
in windows
```
pip install mujoco-python-viewer
python -m mujoco.viewer --mjcf=./Basic_Robot.urdf
# from the viewer, save xml and mfb
```
