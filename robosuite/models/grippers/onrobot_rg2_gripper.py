"""
Gripper for OnRobot's RG2 (has two fingers).

note : 
    this is based on the .xml files which are an unofficial implementation found online and slightly modified, 
    not part of the official robosuite library so careful with namings and functionalities, also, 
    this class is just a copy of the Panda gripper with minor changes.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnRobotRG2GripperBase(GripperModel):
    """
    Gripper for OnRobot's RG2 (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/onrobot_rg2_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            # finger 1
            "left_finger": [
                "finger1_collision",
                "finger1_pad_collision"
            ],

            # finger 2
            "right_finger": [
                "finger2_collision", 
                "finger2_pad_collision"
            ],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class OnRobotRG2Gripper(OnRobotRG2GripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
