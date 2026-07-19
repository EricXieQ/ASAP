#!/usr/bin/env bash
#
# Resume the v5 dab fine-tune from model_6700 (the first v5 run died at ~03:17,
# ~1h45m in, cause unknown because it was launched with >/dev/null -- so this
# time we LOG properly). Same v5 reward config (tightened joint_pos / body_rot
# sigma 1.0->0.25, action_rate -0.5, ankle command penalty kept).
#
# v5@6700 sim-gate results (free-standing, vs v4):
#   waist_roll  +10.2 -> +1.1 deg  (ref +2.0)   FIXED
#   L/R knee    1.7x  -> 1.2x                    FIXED (was downstream of the roll)
#   waist_pitch +2.3  -> -7.8 deg  (ref -21.7)   improving, wants more iters
#   yaw         +8.3  -> +6.1 deg  (ref 0)       stalled -- policy is heading-blind
#   R_ankle_roll        5.9 deg max              ankle fix holding
#
# LAUNCH IN TMUX WITH A LOG (the whole point of this run):
#   tmux new -s dab5
#   conda activate hvgym
#   cd ~/Project/humanoid/ASAP
#   bash finetune_dab_v5_resume.sh 2>&1 | tee /tmp/dab_v5_resume.log
#   # Ctrl+B then D to detach.  Watch: tail -f /tmp/dab_v5_resume.log
#   # `tee` writes the file AND shows it -- so if it dies again, the error is saved.
set -e
set -o pipefail   # so a failure is visible even through the `tee` pipe

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
  experiment_name=DabTracking_spineyaw_v5_resume \
  +checkpoint=logs/DabTracking/20260717_013225-DabTracking_spineyaw_v5-motion_tracking-g1_29dof_anneal_23dof/model_6700.pt \
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
