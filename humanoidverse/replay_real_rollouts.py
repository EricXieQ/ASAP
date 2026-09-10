"""Replay recorded real-robot actions in simulation with the delta action zeroed.

The point: ASAP's delta action model exists to close the gap between what the
simulator does and what the robot actually did, given the same actions. Before
fitting that model it is worth measuring the gap itself -- replay the recorded
actions open-loop, with delta_a forced to zero, and compare per joint against the
recorded real trajectory. Where the error concentrates IS the sim-to-real gap
profile, and it decides 4-DoF (ankle-only) vs full 23-DoF honestly rather than by
inheriting the paper's choice.

The paper narrows to 4 DoF for two stated reasons: (1) limited real data made
23-DoF infeasible -- a data budget, which 473 clips no longer binds -- and (2) the
G1's ankle uses a parallel linkage that simulation models badly. Reason (2) is a
physical argument that was never measured on hardware. This measures it.

Costs no robot time; the data already exists.

Usage (needs LD_LIBRARY_PATH=$CONDA_PREFIX/lib for isaacgym):
  python humanoidverse/replay_real_rollouts.py \
      +exp=train_delta_a_open_loop +robot=g1/g1_29dof_anneal_23dof \
      +simulator=isaacgym +motion_file=/path/to/asap_s2.pkl \
      +num_motions=64 +out=/tmp/replay.npz
"""
import os
import sys
from pathlib import Path

# isaacgym MUST be imported before torch, or its bindings refuse to load.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True,
                    help="a trained run's config.yaml. Using the config the policy was "
                         "actually trained under means the sim physics here match the "
                         "physics the gap is being measured against, and it is fully "
                         "resolved so no hydra group juggling is needed.")
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--num-motions", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=50,
                    help="steps per short rollout. 50 @49Hz ~= 1.0s, the paper's "
                         "closed-loop optimum.")
    ap.add_argument("--batches", type=int, default=20,
                    help="independent reset-and-roll batches. Each reset drops every "
                         "env at a fresh random point in its motion, so the gap is "
                         "sampled across the whole trajectory rather than only from "
                         "frame 0.")
    ap.add_argument("--delta-checkpoint", default=None,
                    help="a trained delta-action model. Without it the delta is forced "
                         "to zero and this measures the RAW sim-to-real gap; with it, "
                         "how much of that gap the model closes. Same protocol either "
                         "way, so the two numbers are directly comparable.")
    ap.add_argument("--seed", type=int, default=None,
                    help="make the scored sample identical across conditions: the same "
                         "clips and the same start times. Without it neither holds. The "
                         "motion library is loaded during env construction with "
                         "random_sample=True and draws via torch.multinomial from an "
                         "unseeded RNG, so each process scores a DIFFERENT random subset "
                         "of the dataset (set_is_evaluating only runs afterwards and does "
                         "not affect that load). Start times then diverge too. Comparing "
                         "two models without this compares them on different data. "
                         "Default None leaves the previous behaviour untouched.")
    ap.add_argument("--out", default="/tmp/replay.npz")
    args = ap.parse_args()

    if args.seed is not None:
        # BEFORE env construction, which is where the motion subset is drawn.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # registers the ${eval:}/${if:}/${sqrt:} resolvers the saved config uses
    import humanoidverse.utils.config_utils  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config

    motion_file = args.motion_file
    n_motions = args.num_motions
    out_path = args.out

    config = OmegaConf.load(args.base_config)
    if config.get("eval_overrides", None) is not None:
        config = OmegaConf.merge(config, config.eval_overrides)

    # --- force the replay condition -----------------------------------------
    OmegaConf.set_struct(config, False)
    config.env._target_ = "humanoidverse.envs.delta_a.delta_a_open_loop.DeltaA_OpenLoop"
    config.robot.motion.motion_file = motion_file
    config.num_envs = n_motions
    config.headless = True
    # zero_delta_a is the baseline condition; with a checkpoint we let the model act.
    config.env.config.zero_delta_a = args.delta_checkpoint is None
    config.env.config.add_extra_action = True   # replay the recorded action
    config.env.config.noise_to_initial_level = 0
    config.env.config.max_episode_length_s = 20
    # The v6 fine-tune starts a fraction of envs from a locomotion stance rather
    # than the motion's own first frame. Great for deploy robustness, fatal here:
    # those envs would be compared against a trajectory they never started on.
    config.env.config.off_trajectory_entry_fraction = 0.0
    config.env.config.resample_motion_when_training = False
    # Deterministic motion selection (is_evaluating below) + randomised start
    # times, so every batch samples a different slice of the trajectory while the
    # SET of motions stays fixed and the numbers are reproducible.
    config.env.config.enforce_randomize_motion_start_eval = True
    config.env.config.termination.terminate_when_motion_far = False
    config.env.config.termination.terminate_by_contact = False
    config.env.config.termination.terminate_by_gravity = False
    config.env.config.termination.terminate_by_low_height = False
    # Domain randomisation would blur exactly the quantity being measured.
    dr = config.env.config.get("domain_rand", None)
    if dr is not None:
        for k in list(dr.keys()):
            if str(k).startswith("randomize") or str(k) in ("push_robots",):
                try:
                    dr[k] = False
                except Exception:
                    pass
        logger.info(f"domain_rand disabled: {[k for k in dr.keys() if str(k).startswith('randomize')]}")

    pre_process_config(config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = instantiate(config.env, device=device)
    env.set_is_evaluating()   # deterministic motion set, not a random 128-of-404
    obs_dict = env.reset_all()

    delta_policy = None
    if args.delta_checkpoint:
        algo = instantiate(config.algo, env=env, device=device, log_dir=None)
        algo.setup()
        algo.load(args.delta_checkpoint)
        algo._eval_mode()
        delta_policy = algo._get_inference_policy()
        logger.info(f"delta model loaded: {args.delta_checkpoint}")
    else:
        logger.info("no delta checkpoint -- measuring the RAW gap (delta = 0)")
    ml = env._motion_lib
    ndof = env.num_dofs

    n_env = env.num_envs
    logger.info(f"replaying {n_env} motions, {args.batches} batches x {args.horizon} "
                f"steps ({args.horizon/49:.2f}s), delta_a=0")

    env_org = getattr(env.simulator, "env_origins", None)
    if env_org is None:
        env_org = torch.zeros(n_env, 3, device=device)
        logger.warning("no env_origins found; root comparison may be offset")
    env_org = env_org.to(device)[:n_env, :3]

    H, B = args.horizon, args.batches
    zero_act = torch.zeros(n_env, env.dim_actions, device=device)

    # (batch, step, env, dof) -- short rollouts from many start points. Free-running
    # for a whole 6s episode conflates the dynamics gap with accumulated integration
    # error (root drifts 138mm by 1s, 865mm by 2s), so the gap is measured over a
    # bounded horizon and re-anchored to the real state each batch.
    sim_dof = np.zeros((B, H, n_env, ndof), np.float32)
    ref_dof = np.zeros((B, H, n_env, ndof), np.float32)
    sim_root = np.zeros((B, H, n_env, 3), np.float32)
    ref_root = np.zeros((B, H, n_env, 3), np.float32)

    if args.seed is not None:
        logger.info(f"seeded with {args.seed} before env construction, and re-seeded "
                    f"per batch: same clips and same start times across conditions")

    for b in range(B):
        if args.seed is not None:
            # Per batch, not once before the loop. Seeding once is not enough: envs
            # whose random t0 lands near the end of a clip hit terminate_when_motion_end
            # mid-horizon and resample, and the rollout itself may draw, so by batch 2
            # the conditions have consumed different amounts of RNG. Re-seeding here
            # makes the start times a pure function of (seed, batch index) and of
            # nothing the policy did.
            torch.manual_seed(args.seed + b)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + b)
        obs_dict = env.reset_all()   # re-anchors every env to the real state at a new t0
        for h in range(H):
            times = env.episode_length_buf * env.dt + env.motion_start_times
            st = ml.get_motion_state(env.motion_ids, times)
            ref_dof[b, h] = st["dof_pos"].cpu().numpy()
            ref_root[b, h] = st["root_pos"].cpu().numpy()
            sim_dof[b, h] = env.simulator.dof_pos.cpu().numpy()
            sim_root[b, h] = (env.simulator.robot_root_states[:, :3] - env_org).cpu().numpy()
            if delta_policy is not None:
                act = delta_policy(obs_dict["actor_obs"]).detach()
            else:
                act = zero_act
            obs_dict, _, _, _ = env.step({"actions": act})
        if (b + 1) % 5 == 0:
            e = np.degrees(np.abs(sim_dof[b] - ref_dof[b]))
            logger.info(f"  batch {b+1}/{B}: mean {e.mean():.2f} deg at horizon {H}")

    alive = np.ones(sim_dof.shape[:3], np.float32)
    np.savez_compressed(
        out_path, sim_dof=sim_dof, ref_dof=ref_dof,
        sim_root=sim_root, ref_root=ref_root, alive=alive,
        dof_names=np.array(env.simulator.dof_names if hasattr(env.simulator, "dof_names") else []),
    )
    e = np.degrees(np.abs(sim_dof - ref_dof))
    logger.info(f"wrote {out_path}")
    tag = "WITH delta model" if delta_policy is not None else "baseline (delta = 0)"
    logger.info(f"  condition: {tag}")
    for frac in (0.25, 0.5, 1.0):
        k = min(int(frac * 49), H - 1)
        logger.info(f"  open-loop MPJPE @{frac:.2f}s : {e[:, k].mean():.3f} deg")


if __name__ == "__main__":
    main()
