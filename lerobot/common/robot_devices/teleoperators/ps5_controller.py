# robot: KochRobot = make_robot("lerobot/configs/robot/koch_.yaml")
# robot.follower_arms.
import threading
import time

import inputs
import numpy as np

from lerobot.common.kinematics import KochKinematics
from lerobot.common.robot_devices.robots.koch import KochRobot

EE_SENSITIVITY = 5e-5
JOINT_SENSITIVITY = 0.04


class PS5Controller:
    def __init__(self, robot: KochRobot):
        try:
            inputs.devices.gamepads[0]
        except IndexError as e:
            raise inputs.UnpluggedError("No gamepad found.") from e
        self.gamepad_last_state = [128, 128, 128, 128, 0, 0]
        self.gamepad_read_thread = threading.Thread(target=self.read_gamepad)
        self.gamepad_read_thread.start()

    def read_gamepad(self):
        while True:
            events = inputs.get_gamepad()
            for event in events:
                if event.code == "ABS_X":  # Left stick X-axis
                    self.gamepad_last_state[0] = event.state
                if event.code == "ABS_Y":  # Left stick Y-axis
                    self.gamepad_last_state[1] = event.state
                if event.code == "ABS_RX":  # Right stick X-axis
                    self.gamepad_last_state[2] = event.state
                if event.code == "ABS_RY":  # Right stick Y-axis
                    self.gamepad_last_state[3] = event.state
                if event.code == "ABS_RZ":  # right trigger
                    self.gamepad_last_state[4] = event.state
                if event.code == "ABS_Z":  # left trigger
                    self.gamepad_last_state[5] = event.state

            time.sleep(0.0001)

    def read(self, reference_joint_state):
        left_right = self.gamepad_last_state[0] - 128
        forward_backward = self.gamepad_last_state[1] - 128
        up = self.gamepad_last_state[4]
        down = self.gamepad_last_state[5]
        wrist_flex = self.gamepad_last_state[3] - 128
        wrist_twist = self.gamepad_last_state[2] - 128
        print(self.gamepad_last_state)
        if abs(left_right) <= 10:
            left_right = 0
        if abs(forward_backward) <= 10:
            forward_backward = 0
        if abs(wrist_flex) <= 10:
            wrist_flex = 0
        if abs(wrist_twist) <= 10:
            wrist_twist = 0
        if abs(up) <= 10:
            up = 0
        if abs(down) <= 10:
            down = 0

        if (
            left_right == 0
            and forward_backward == 0
            and wrist_flex == 0
            and wrist_twist == 0
            and up == 0
            and down == 0
        ):
            return None

        current_ee_pose = KochKinematics.fk_gripper(reference_joint_state)

        # Compute IK
        if left_right != 0 or forward_backward != 0 or up != 0 or down != 0:
            print("IK")
            desired_ee_pose = (
                np.array(
                    [
                        [1, 0, 0, -left_right * EE_SENSITIVITY],
                        [0, 1, 0, forward_backward * EE_SENSITIVITY],
                        [0, 0, 1, (up - down) * EE_SENSITIVITY / 2],
                        [0, 0, 0, 1],
                    ]
                )
                @ current_ee_pose
            )
            target_joint_state = KochKinematics.ik(
                reference_joint_state, desired_ee_pose, position_only=left_right != 0
            )
            if target_joint_state is None:
                raise AssertionError
        else:
            target_joint_state = reference_joint_state.copy()

        if wrist_flex != 0:
            target_joint_state[3] += wrist_flex * JOINT_SENSITIVITY

        if wrist_twist != 0:
            target_joint_state[4] += wrist_twist * JOINT_SENSITIVITY

        print(target_joint_state)

        return target_joint_state
