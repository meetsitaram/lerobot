import cv2
import numpy as np
import torch

from lerobot.common.kinematics import KochKinematics
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.vision import segment_hsv

import time
import json


GRIPPER_TIP_Z_BOUNDS = (0.005, 0.06)
GRIPPER_TIP_X_BOUNDS = (-0.16, 0.16)
GRIPPER_TIP_Y_BOUNDS = (-0.25, -0.06)
GRIPPER_TIP_BOUNDS = np.row_stack([GRIPPER_TIP_X_BOUNDS, GRIPPER_TIP_Y_BOUNDS, GRIPPER_TIP_Z_BOUNDS])


def is_in_bounds(gripper_tip_pos, buffer: float | np.ndarray = 0):
    if not isinstance(buffer, np.ndarray):
        buffer = np.zeros_like(GRIPPER_TIP_BOUNDS) + buffer
    for i, bounds in enumerate(GRIPPER_TIP_BOUNDS):
        assert (bounds[1] - bounds[0]) > buffer[i].sum()
        if gripper_tip_pos[i] < bounds[0] + buffer[i][0] or gripper_tip_pos[i] > bounds[1] - buffer[i][1]:
            return False
    return True


def calc_smoothness_reward(
    action: np.ndarray,
    prior_action: np.ndarray | None = None,
    first_order_coeff: float = -1.0,
    second_order_coeff: float = -1.0,
):
    reward = first_order_coeff * np.linalg.norm(action)
    if prior_action is not None:
        reward += second_order_coeff * np.linalg.norm(action - prior_action)
    return reward


def calc_reward_cube_push(
    img,
    goal_mask,
    current_joint_pos,
    distance_reward_coeff: float = 1 / 24,
    action: np.ndarray | None = None,
    prior_action: np.ndarray | None = None,
    first_order_smoothness_coeff: float = -0,
    second_order_smoothness_coeff: float = -0.02,
    oob_reward: float = -5.0,
    occlusion_limit=55,
    occlusion_reward=-3.0,
) -> tuple[float, bool, bool, dict]:
    obj_mask, annotated_img = segment_hsv(img)

    if np.count_nonzero(obj_mask) >= occlusion_limit:
        intersection_area = np.count_nonzero(np.bitwise_and(obj_mask, goal_mask))

        success = False
        if intersection_area <= occlusion_limit:
            # Find the minimum distance between the object and the goal.
            goal_contour = cv2.findContours(
                goal_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )[0]
            obj_contour = cv2.findContours(
                obj_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )[0]
            obj_points = np.vstack(obj_contour).squeeze()  # shape (N, 2)
            goal_points = np.vstack(goal_contour).squeeze()  # shape (M, 2)
            distances = np.linalg.norm(obj_points[:, None] - goal_points[None, :], axis=-1)  # shape (N, M, 2)
            reward = -np.min(distances) * distance_reward_coeff
        elif intersection_area > 0:
            reward = intersection_area / np.count_nonzero(obj_mask)
            success = reward == 1
    else:
        success = False
        reward = occlusion_reward

    do_terminate = False

    # gripper_tip_pos = KochKinematics.fk_gripper_tip(current_joint_pos)[:3, -1]
    # if not is_in_bounds(gripper_tip_pos):
    #     do_terminate = True
    #     reward += oob_reward

    if success:
        do_terminate = True
        reward += 5

    if action is not None:
        reward += calc_smoothness_reward(
            action, prior_action, first_order_smoothness_coeff, second_order_smoothness_coeff
        )

    # Lose 1 for each step to encourage faster completion.
    reward -= 1

    info = {"annotated_img": annotated_img}

    return reward, success, do_terminate, info


def calc_reward_joint_goal(
    current_joint_pos,
    action: np.ndarray | None = None,
    prior_action: np.ndarray | None = None,
    first_order_smoothness_coeff: float = -1.0,
    second_order_smoothness_coeff: float = -1.0,
):
    # Whole arm
    goal = np.array([87, 82, 91, 65, 3, 30])
    curr = current_joint_pos
    reward = -np.abs(goal - curr).mean() / 10
    success = np.abs(goal - curr).max() <= 3

    do_terminate = False

    gripper_tip_pos = KochKinematics.fk_gripper_tip(current_joint_pos)[:3, -1]
    if not is_in_bounds(gripper_tip_pos):
        reward -= 5
        do_terminate = True

    if success:
        do_terminate = True
        reward += 1

    if action is not None:
        reward += calc_smoothness_reward(
            action, prior_action, first_order_smoothness_coeff, second_order_smoothness_coeff
        )

    return reward, success, do_terminate


def _go_to_pos(robot, pos, tol=None):

    pos = robot.follower_arms["main"].read("Present_Position")
    print ("robot current position:", pos)
    time.sleep(2)
    return

    if tol is None:
        tol = np.array([3, 3, 3, 10, 3, 3])
    while True:
        robot.send_action(pos)
        current_pos = robot.follower_arms["main"].read("Present_Position")
        busy_wait(1 / 30)
        if np.all(np.abs(current_pos - pos.numpy()) < tol):
            break


def reset_for_joint_pos(robot: ManipulatorRobot):
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    reset_pos = robot.follower_arms["main"].read("Present_Position")
    while True:
        # For reference: goal = [87, 82, 91, 65, 3, 30]
        reset_pos[0] = np.random.uniform(70, 110)
        reset_pos[1] = np.random.uniform(70, 90)
        reset_pos[2] = np.random.uniform(80, 110)
        reset_pos[3] = np.random.uniform(45, 90)
        reset_pos[4] = np.random.uniform(-50, 50)
        reset_pos[5] = np.random.uniform(0, 90)
        if is_in_bounds(KochKinematics.fk_gripper_tip(reset_pos)[:3, -1], buffer=0.02):
            break
    reset_pos = torch.from_numpy(reset_pos)
    _go_to_pos(robot, reset_pos)

def get_max_boundaries(robot: ManipulatorRobot):
    print("getting max bounds")
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)

    bounds = []
    for i in range(0,8):
        print(f"sample position {i}")
        time.sleep(4)
        pos = robot.follower_arms["main"].read("Present_Position")
        bounds.append(pos.tolist())
    
    for pos_idx, pos in enumerate(bounds):
        for idx,val in enumerate(pos):
            bounds[pos_idx][idx] = round(pos[idx])

    return bounds


def reset_for_cube_push(robot: ManipulatorRobot, right=True):
    # robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)

    staging_pos = torch.tensor([90, 100, 60, 65, 3, 30]).float()
    while True:
        reset_pos = torch.tensor(
            [
                np.random.uniform(125, 135) if right else np.random.uniform(45, 55),
                np.random.uniform(54, 58),
                np.random.uniform(50, 52),
                np.random.uniform(78, 98),
                np.random.uniform(-41, -31) if right else np.random.uniform(31, 41),
                np.random.uniform(0, 20),
            ]
        ).float()
        if is_in_bounds(
            KochKinematics.fk_gripper_tip(reset_pos.numpy())[:3, -1],
            buffer=np.array([[0.02, 0.02], [0.02, 0.02], [0.02, 0.01]]),
        ):
            break
    intermediate_pos = torch.from_numpy(robot.follower_arms["main"].read("Present_Position"))
    intermediate_pos[1] = staging_pos[1]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 5, 5, 10, 5, 5]))
    intermediate_pos[1:] = staging_pos[1:]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 5, 5, 10, 5, 5]))
    if right and staging_pos[0] > intermediate_pos[0]:  # noqa: SIM114
        _go_to_pos(robot, staging_pos, tol=np.array([5, 5, 5, 10, 5, 5]))
    elif (not right) and staging_pos[0] < intermediate_pos[0]:
        _go_to_pos(robot, staging_pos, tol=np.array([5, 5, 5, 10, 5, 5]))
    intermediate_pos = staging_pos.clone()
    intermediate_pos[0] = reset_pos[0]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 5, 5, 10, 5, 5]))
    _go_to_pos(robot, reset_pos)

def tmp_save_boundaries(left_goal_bounds, right_goal_bounds, max_bounds ):
    left_goal_bounds = [[-18.720703125, 111.005859375, 131.30859375, 67.587890625, 256.2890625, 1.142578125], [-10.283203125, 66.62109375, 80.68359375, 75.673828125, 258.837890625, 0.87890625], [-35.947265625, 48.251953125, 49.833984375, 81.474609375, 46.0546875, 1.0546875], [-59.23828125, 76.46484375, 96.416015625, 68.818359375, 28.30078125, 0.791015625], [-12.83203125, 118.30078125, 116.71875, 85.869140625, -110.830078125, 1.23046875], [-7.55859375, 75.498046875, 69.43359375, 87.451171875, -100.458984375, 1.23046875], [-36.826171875, 52.119140625, 37.529296875, 93.076171875, 49.74609375, 0.87890625], [-58.095703125, 83.14453125, 83.671875, 83.49609375, 38.056640625, 0.87890625]]
    right_goal_bounds = [[48.8671875, 89.82421875, 113.203125, 63.896484375, -51.328125, 0.791015625], [32.51953125, 60.556640625, 71.982421875, 75.498046875, -59.150390625, 1.142578125], [0.087890625, 70.224609375, 88.154296875, 72.685546875, 82.705078125, 0.966796875], [-4.921875, 116.015625, 133.681640625, 64.16015625, 79.541015625, 1.23046875], [51.50390625, 95.625, 94.21875, 83.14453125, -46.0546875, 1.142578125], [33.486328125, 64.16015625, 52.294921875, 94.74609375, -60.29296875, 1.318359375], [-1.494140625, 81.03515625, 79.27734375, 84.0234375, 80.068359375, 1.142578125], [-6.6796875, 118.125, 115.224609375, 85.341796875, 76.81640625, 1.142578125]]
    max_bounds = [[71.71875, 92.900390625, 116.54296875, 56.162109375, -7.998046875, 1.142578125], [31.904296875, 45.52734375, 45.791015625, 71.806640625, -39.287109375, 1.845703125], [-35.244140625, 31.376953125, 18.28125, 87.5390625, 81.2109375, 1.142578125], [-79.013671875, 80.244140625, 102.744140625, 59.150390625, 57.3046875, 0.615234375], [77.783203125, 96.591796875, 90.263671875, 88.76953125, -13.974609375, 0.615234375], [32.87109375, 48.1640625, 25.576171875, 89.82421875, -60.29296875, 0.615234375], [-37.08984375, 41.220703125, 15.1171875, 84.814453125, 18.10546875, 0.703125], [-78.486328125, 80.419921875, 76.904296875, 86.484375, 9.931640625, 0.703125]]

    for pos_idx, pos in enumerate(left_goal_bounds):
        for idx,val in enumerate(pos):
            left_goal_bounds[pos_idx][idx] = round(pos[idx])

    for pos_idx, pos in enumerate(right_goal_bounds):
        for idx,val in enumerate(pos):
            right_goal_bounds[pos_idx][idx] = round(pos[idx])

    for pos_idx, pos in enumerate(max_bounds):
        for idx,val in enumerate(pos):
            max_bounds[pos_idx][idx] = round(pos[idx])

    boundaries = {
        'left_goal_bounds':left_goal_bounds,
        'right_goal_bounds':right_goal_bounds,
        # 'center_bounds': center_bounds,
        'max_bounds': max_bounds
    }

    with open('calib_max_boundaries.json', 'w') as f:
        json.dump(boundaries, f)


if __name__ == "__main__":

    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.utils.utils import init_hydra_config

    robot = make_robot(init_hydra_config("lerobot/configs/robot/koch-my.yaml"))

    print("connecting robot")
    robot.connect()
    
    print('get left goal boundaries')
    left_goal_bounds = get_max_boundaries(robot)
    print(left_goal_bounds)

    print('get right goal boundaries')
    right_goal_bounds = get_max_boundaries(robot)
    print(right_goal_bounds)

    # print('get centre boundaries')
    # center_bounds = get_max_boundaries(robot)
    # print(center_bounds)

    print('get max boundaries')
    max_bounds = get_max_boundaries(robot)
    print(max_bounds)

    boundaries = {
        'left_goal_bounds':left_goal_bounds,
        'right_goal_bounds':right_goal_bounds,
        # 'center_bounds': center_bounds,
        'max_bounds': max_bounds
    }

    with open('calib_max_boundaries.json', 'w') as f:
        json.dump(boundaries, f)

    print("done calibrating max boundaries")

    # print("reset for cube push to left")
    # reset_for_cube_push(robot, right=False)

    robot.disconnect()
