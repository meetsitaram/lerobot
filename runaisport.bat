#python lerobot/scripts/control_robot.py record --robot-path lerobot/configs/robot/koch.yaml --fps 30 --root data --repo-id workshop/eval_koch_sport --tags tutorial --push-to-hub 0 --warmup-time-s 45 --episode-time-s 60 --reset-time-s 15 --num-episodes 1 -p outputs/train/koch_sport/checkpoints/last/pretrained_model

python evaluate.py