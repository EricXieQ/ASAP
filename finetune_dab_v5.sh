#!/usr/bin/env bash
#
# Fine-tune v5: fix the crooked spine + yaw drift + left-knee over-work.
#
# Warm-starts from v4's model_41600 (which ALREADY fixed the R_ankle_roll command
# wind-up: 109.8 deg -> 5.7 deg, verified in rl_sim_mujoco). v5 keeps that and adds:
#
#   reward_motion_tracking_dm_2real_v5:
#     - teleop_joint_pos sigma  1.0 -> 0.25   (enforce the spine/joint pose)
#     - teleop_body_rot  sigma  1.0 -> 0.25   (enforce heading -> yaw drift)
#     - teleop_body_rotation_extend weight back to 0.5 (v4 proved weight is the
#       wrong lever; the sigma sharpens the gradient)
#     - penalty_action_rate back to -0.5 (the -0.25 fed the left-knee jitter;
#       snappiness comes from running at 100% gains, not from this term)
#     - penalty_dof_pos_command_over_limit -10.0 KEPT (the ankle fix)
#
# WHY: rewards are exp(-err^2/sigma). At sigma=1.0 a 24 deg (0.42 rad) spine error
# still retains 84% of max reward -> no gradient. At 0.25 it drops to 50%.
#
# RUN IT IN TMUX (a display/X crash killed the v4 run at 11h; tmux survives that),
# and redirect output to a file so the terminal isn't GPU-rendering a firehose of
# text on the same 4090 that's training:
#
#   tmux new -s dab5
#   conda activate hvgym
#   cd ~/Project/humanoid/ASAP
#   bash finetune_dab_v5.sh > /tmp/dab_v5.log 2>&1
#   # Ctrl+B then D to detach.  Reattach: tmux attach -t dab5
#   # Peek at progress:        tail -f /tmp/dab_v5.log
set -e

# IsaacGym's compiled .so links against libpython3.8.so.1.0 in the conda env.
export LD_LIBRARY_PATH="/home/eric/miniconda3/envs/hvgym/lib:${LD_LIBRARY_PATH}"

python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +robot=g1/g1_29dof_anneal_23dof \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real_v5 \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  +domain_rand=domain_rand_base \
  +terrain=terrain_locomotion_plane \
  num_envs=4096 \
  algo.config.num_learning_iterations=100000 \
  project_name=DabTracking \
  experiment_name=DabTracking_spineyaw_v5 \
  +checkpoint=logs/DabTracking/20260716_131145-DabTracking_ankleyaw_v4-motion_tracking-g1_29dof_anneal_23dof/model_41600.pt \
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

# --- Verify a checkpoint BEFORE deploying (export -> rl_sar -> sim gate):
#   python rl_sar/policy/g1/asap_dab/export_wrapper.py --ckpt logs/DabTracking/<v5-run>/model_XXXXX.pt
#   cd rl_sar && RL_TRACE=1 ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof
#   keys: 0 -> 1 -> R (reset, required for no-hoist) -> '+'x40 -> ']'x40 -> 5 (dab)
# Check /tmp/rl_sar_trace.csv (test FREE-STANDING; hoist2's ropes mask the yaw):
#   - R_ankle_roll target within +-15 deg          (v4 baseline: 5.7 -- must not regress)
#   - waist_roll mean near ref +2.0 deg            (v4 was +10.2 = crooked spine)
#   - waist_pitch mean near ref -21.7 deg          (v4 was +2.3 = 24 deg off)
#   - per-dab yaw near 0                           (v4 free-standing: +8.3/dab, net +93)
#   - L vs R knee range/jitter comparable          (v4: left ~1.7x busier)
