"""Carts2Pole articulation configuration.

Defines the dual-cart + pole (Carts2Pole) model configuration used for
Isaac Lab simulation, including asset loading, initial state, and joint
actuator parameters.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# CartPole robot configuration object
CARTPOLE_CFG = ArticulationCfg(
    # Load the model from a USD file and set its physics properties
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu22/tdmpc/src/assets/carts2pole.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    # Initial state of the robot
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 4.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            "slider_to_cart1": 0.0,  # initial position between slider and cart1
            "cart_to_cart1": 0.0,    # initial spring extension (matches USD target position)
            "cart_to_pole": 0.0,     # initial pole angle
        },
        joint_vel={
            "slider_to_cart1": 0.0,
            "cart_to_cart1": 0.0,
            "cart_to_pole": 0.0,
        },
    ),
    # Joint actuators
    actuators={
        # cart1 slider joint (actively controlled)
        "cart1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart1"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # pole hinge (passive)
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)
