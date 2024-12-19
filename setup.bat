del .cache\calibration\koch\main_follower.json
del .cache\calibration\koch\main_leader.json
del /q data\workshop
del /q outputs\train
python lerobot/scripts/control_robot.py teleoperate --robot-path lerobot/configs/robot/koch.yaml --robot-overrides ~cameras