#!/usr/bin/env bash
#
# Fine-tune v6: reduce the LEFT-LEG PIVOT / entry stomp.
#
# Insight (Eric): the robot NEVER enters the dab mid-air, tilted, or dropping --
# it always enters from a STANDING loco stance, both feet planted. But v3/v4/v5
# trained with noise_to_initial_level=2, which spawns the robot at a random
# HEIGHT (line 475, root_pos[:3] incl Z), TILT (root_rot), and VELOCITY
# (root_vel/root_ang_vel). So 20%+ of episodes start dropped/tumbling and the
# policy learned a left-leg pivot to catch itself -- a skill it never uses at
# deploy, but which fires on every (clean) entry as the visible "stomp".
#
# FIX: keep the REALISTIC entry, cut the UNREALISTIC perturbation.
#   - KEEP off_trajectory_entry_fraction=0.2  (enter from the standing loco
#     stance -- this IS the real FSM-switch scenario, worth training)
#   - noise_to_initial_level 2 -> 0.25  (stop dropping/tilting/tumbling the
#     robot at spawn; it starts ~clean & upright like real life)
#
# Warm-starts from v5's model_13000 (ankle fixed, spine fixed, knees balanced).
# The reduced init noise should also make training STABLER than v5 (fewer extreme
# states = less chance of the NaN blowup that killed the v5 run).
#
# Bonus: fewer extreme states may also let this converge without the reward
# cliffs v5 showed. Still, RUN IT IN TMUX WITH A LOG:
#   tmux new -s dab6
#   conda activate hvgym
#   cd ~/Project/humanoid/ASAP
#   bash finetune_dab_v6.sh 2>&1 | tee /tmp/dab_v6.log
#   # Ctrl+B then D to detach.  Watch: tail -f /tmp/dab_v6.log
set -e
set -o pipefail

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
  experiment_name=DabTracking_grounded_v6 \
  +checkpoint=logs/DabTracking/20260717_124339-DabTracking_spineyaw_v5_resume-motion_tracking-g1_29dof_anneal_23dof/model_13000.pt \
  robot.motion.motion_file=humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_wall_background_dab_amass.pkl \
  env.config.noise_to_initial_level=0.25 \
  env.config.init_noise_scale.root_rot=2 \
  env.config.termination_scales.termination_gravity_x=0.95 \
  env.config.termination_scales.termination_gravity_y=0.95 \
  env.config.termination.terminate_by_low_height=True \
  env.config.termination_scales.termination_min_base_height=0.3 \
  +env.config.off_trajectory_entry_fraction=0.2 \
  algo.config.actor_learning_rate=1.e-4 \
  algo.config.critic_learning_rate=2.e-4 \
  +algo.config.max_learning_rate=3.e-4 \
  algo.config.max_grad_norm=0.5 \
  headless=True

# GRADIENT-EXPLOSION SAFEGUARDS (v5 died on a NaN at ~30k iters):
#   1. NaN guard in ppo.py: a non-finite loss/grad SKIPS that update instead of
#      corrupting the policy -> training survives a bad batch.
#   2. max_learning_rate=3e-4: caps the adaptive LR so it can't ratchet toward
#      1e-2 (the likely v5 cause). Still adaptive, just bounded.
#   3. max_grad_norm 1.0 -> 0.5: tighter gradient clipping.
#   4. (v6 also spawns the robot ~clean, not dropped/tumbling -> fewer extreme
#      states -> fewer chances for a blow-up in the first place.)

# --- Verify (export -> rl_sar -> free-standing gate):
#   python rl_sar/policy/g1/asap_dab/export_wrapper.py --ckpt logs/DabTracking/<v6-run>/model_XXXXX.pt
#   cd rl_sar && RL_TRACE=1 ./cmake_build/bin/rl_sim_mujoco g1 scene_29dof   # 0 1 R +x40 ]x40 5
# Check: left-knee ENTRY jerk (was ~11.5 deg/tick) should DROP; ankle still <=15;
#        spine/yaw not regressed. The stomp should visibly soften on clean entries.
