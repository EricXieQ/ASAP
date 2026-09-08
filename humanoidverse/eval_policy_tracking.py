"""Score a tracking policy by how closely it reproduces the REAL recorded trajectory.

The motion reference here is real robot data, so tracking error against it is a
direct measure of "does this policy move like the real robot did". Run the same
policy in the uncorrected simulator and in one corrected by a frozen delta model,
and the pair separates two things that are easy to conflate: how good the policy
is, and how realistic the simulator is.

Reward cannot answer this. Each fine-tuning arm trained inside a different
simulator, so their rewards are computed under different dynamics and are not
comparable across arms. The policies have to be brought into a common environment.

  python humanoidverse/eval_policy_tracking.py \
      --base-config <a finetune run>/config.yaml \
      --policy-checkpoint <run>/model_3000.pt \
      --motion-file asap_dab_473.pkl \
      [--delta-checkpoint <delta run>/model_5000.pt]   # omit for the plain sim
"""
import sys
from pathlib import Path

try:
    import isaacgym  # noqa: F401  (must precede torch)
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
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--policy-checkpoint", required=True)
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--delta-checkpoint", default=None,
                    help="corrects the simulator during EVALUATION. Omit to score "
                         "the policy in the plain simulator.")
    ap.add_argument("--num-motions", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import humanoidverse.utils.config_utils  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config

    config = OmegaConf.load(args.base_config)
    if config.get("eval_overrides", None) is not None:
        config = OmegaConf.merge(config, config.eval_overrides)
    OmegaConf.set_struct(config, False)

    config.robot.motion.motion_file = args.motion_file
    config.num_envs = args.num_motions
    config.headless = True
    config.env.config.delta_checkpoint = args.delta_checkpoint   # None -> plain sim
    config.env.config.noise_to_initial_level = 0
    config.env.config.resample_motion_when_training = False
    config.env.config.enforce_randomize_motion_start_eval = True
    config.env.config.off_trajectory_entry_fraction = 0.0
    for k in ("terminate_when_motion_far", "terminate_by_contact",
              "terminate_by_gravity", "terminate_by_low_height"):
        config.env.config.termination[k] = False
    dr = config.env.config.get("domain_rand", None)
    if dr is not None:
        for k in list(dr.keys()):
            if str(k).startswith("randomize") or str(k) == "push_robots":
                try:
                    dr[k] = False
                except Exception:
                    pass

    pre_process_config(config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = instantiate(config.env, device=device)
    env.set_is_evaluating()
    obs_dict = env.reset_all()

    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(args.policy_checkpoint)
    algo._eval_mode()
    policy = algo._get_inference_policy()

    ml = env._motion_lib
    H, B, N = args.horizon, args.batches, env.num_envs
    sim_dof = np.zeros((B, H, N, env.num_dofs), np.float32)
    ref_dof = np.zeros_like(sim_dof)

    for b in range(B):
        obs_dict = env.reset_all()
        for h in range(H):
            t = env.episode_length_buf * env.dt + env.motion_start_times
            st = ml.get_motion_state(env.motion_ids, t)
            ref_dof[b, h] = st["dof_pos"].cpu().numpy()
            sim_dof[b, h] = env.simulator.dof_pos.cpu().numpy()
            with torch.no_grad():
                act = policy(obs_dict["actor_obs"]).detach()
            obs_dict, _, _, _ = env.step({"actions": act})

    e = np.degrees(np.abs(sim_dof - ref_dof))
    sim_kind = "delta-corrected" if args.delta_checkpoint else "plain"
    logger.info(f"RESULT {args.label} | policy={Path(args.policy_checkpoint).parent.name} "
                f"| sim={sim_kind}")
    for frac in (0.25, 0.5, 1.0):
        k = min(int(frac * 49), H - 1)
        logger.info(f"  tracking error @{frac:.2f}s : {e[:, k].mean():.3f} deg")
    print(f"CSVROW,{args.label},{sim_kind},{e[:, 12].mean():.4f},{e[:, -1].mean():.4f}")


if __name__ == "__main__":
    main()
