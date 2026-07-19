#!/usr/bin/env bash
#
# Fine-tune v4: fix the R_ankle_roll command wind-up + the clockwise per-dab yaw
# drift, on top of the off-trajectory-entry work from v3.
#
# What's different from finetune_dab.sh (v3):
#   - +rewards = ..._ankleyaw  (adds penalty_dof_pos_command_over_limit,
#                               strengthens teleop_body_rotation_extend 0.5->1.0)
#   - wider friction randomization (0.5..1.25 -> 0.3..1.5) so the policy learns to
#     reject the asymmetric foot-contact yaw moment that the REAL robot amplifies ~3x.
#     (control-delay randomization is already on in domain_rand_base.)
#
# Run from the ASAP repo root with the hvgym conda env active:
#   conda activate hvgym
#   cd ~/Project/humanoid/ASAP
#   bash finetune_dab_v4.sh
#
# Checkpoints save periodically under logs/DabTracking/<...>_ankleyaw_v4/. Verify
# an intermediate checkpoint in rl_sim_mujoco (see the sim-verify command at the
# bottom) BEFORE deploying to the robot -- you can stop early once the ankle target
# stays <=15 deg and per-dab yaw drift is near 0.
set -e

# IsaacGym's compiled .so links against libpython3.8.so.1.0 in the conda env.
export LD_LIBRARY_PATH="/home/eric/miniconda3/envs/hvgym/lib:${LD_LIBRARY_PATH}"

# SLUGGISHNESS: penalty_action_rate is already halved (-0.5 -> -0.25) in the
# ankleyaw reward for a snappier dab. If it's STILL too sluggish, add this override
# to also cut control-delay randomization (0..2 steps ~90ms -> 0..1 ~45ms):
#     domain_rand.ctrl_delay_step_range="[0, 1]" \
# WARNING: ctrl_delay is the policy's defense against the Jetson's timing jitter --
# cutting it can bring back the balance/oscillation sensitivity. Test on the hoist first.
#
# NOTE: +checkpoint points at the deployed off-traj policy (model_25600) so v4 builds
# on the entry work. Swap it if a different checkpoint is live.
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real_ankleyaw \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  +domain_rand=domain_rand_base \
  +terrain=terrain_locomotion_plane \
  num_envs=4096 \
  algo.config.num_learning_iterations=100000 \
  project_name=DabTracking \
  experiment_name=DabTracking_ankleyaw_v4 \
  +checkpoint=logs/DabTracking/20260504_015435-DabTracking_wall_offtraj_finetune_v3-motion_tracking-g1_29dof_anneal_23dof/model_25600.pt \
  robot.motion.motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_wall_background_dab_amass.pkl \
  env.config.noise_to_initial_level=2 \
  env.config.init_noise_scale.root_rot=2 \
  env.config.termination_scales.termination_gravity_x=0.95 \
  env.config.termination_scales.termination_gravity_y=0.95 \
  env.config.termination.terminate_by_low_height=True \
  env.config.termination_scales.termination_min_base_height=0.3 \
  +env.config.off_trajectory_entry_fraction=0.2 \
  algo.config.actor_learning_rate=1.e-4 \
  algo.config.critic_learning_rate=2.e-4 \
  headless=True

# --- After training: export the checkpoint, port to rl_sar policy.pt, then verify:
#   RL_TRACE=1 ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof_hoist   # drive 0 1 5 5 5
# Check /tmp/rl_sar_trace.csv:
#   - R_ankle_roll TARGET (q11) stays within +-15 deg during asap_dab rows
#   - per-dab base yaw drift (quat_w..quat_z) near 0
#   - left/right foot-load symmetry improved (does the left stomp self-correct?)
