"""Fine-tune a motion-tracking policy inside a simulator corrected by a frozen
delta action model.

This is stage (c) of ASAP's pipeline -- "freeze the delta action model, incorporate
it into the simulator to align the real-world physics, and then fine-tune the
pre-trained motion tracking policy". The open-source release does not implement it:
`delta_a_closed_loop.py` reads `self.loaded_extra_policy` in two places but nothing
in the repository ever assigns it, so both branches are dead behind a `hasattr`.
The configs that would feed the stage exist (NO_domain_rand_finetune_with_deltaA,
reward_motion_tracking_use_deltaA_to_train_2real) with no code path behind them.

What happens here: the PPO actor being trained is the POLICY. Each control step its
action is passed through the frozen delta model, and the sum drives the PD
controller -- so the policy experiences a simulator whose dynamics have been pulled
toward the real robot's. The delta model's weights never move.

    torque = kp * ((a_policy + a_delta) * action_scale + q_default - q) - kd * qd

Two details that are easy to get wrong:

1. The delta model's observation must be rebuilt exactly as it was during ITS
   training -- same components, same per-component scales, and concatenated in
   SORTED key order (see `_post_config_observation_callback`), not the order the
   config happens to list them in. We read the component list and scales from the
   delta run's own config.yaml rather than hardcoding, so this tracks whatever the
   model was actually trained with.

2. The two action slots swap meaning. During delta training the PPO actor was the
   delta, so `actions` was the delta's own previous output and `actions_open_loop`
   was the recorded action being corrected. Here the policy supplies the action
   being corrected, so `actions_open_loop` <- the policy's action this step, and
   `actions` <- the delta's own previous output, which we track separately.

Observation noise is NOT applied to the delta's input. It is a frozen model of
physics being queried at inference, not a policy that needs robustness training.
"""
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking

# indices of the four ankle DoF in the 23-DoF ASAP ordering
ANKLE_IDX = [4, 5, 10, 11]


class MotionTrackingDeltaFinetune(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.delta_actor = None
        self.last_delta_action = torch.zeros(
            self.num_envs, self.dim_actions, dtype=torch.float32, device=self.device)
        ckpt = self.config.get("delta_checkpoint", None)
        if ckpt:
            self._load_delta_model(ckpt)
        else:
            logger.warning("no delta_checkpoint given -- this is a plain fine-tune "
                           "with an UNcorrected simulator (useful as the control arm)")

    # ------------------------------------------------------------------ loading
    def _load_delta_model(self, ckpt_path):
        from humanoidverse.agents.modules.ppo_modules import PPOActor

        from humanoidverse.utils.helpers import pre_process_config

        ckpt_path = Path(ckpt_path)
        cfg_path = ckpt_path.parent / "config.yaml"
        if not cfg_path.exists():
            cfg_path = ckpt_path.parent.parent / "config.yaml"
        dcfg = OmegaConf.load(cfg_path)
        # algo_obs_dim_dict is saved as '???' and filled at runtime from the obs
        # config, so the delta run's own config has to be pre-processed before its
        # network dims can be read.
        pre_process_config(dcfg)

        # Built from the DELTA run's own dims, not this env's -- the two have
        # different observations, so instantiating through the algo would fail.
        self.delta_actor = PPOActor(
            obs_dim_dict=dcfg.robot.algo_obs_dim_dict,
            module_config_dict=dcfg.algo.config.module_dict.actor,
            num_actions=dcfg.robot.actions_dim,
            init_noise_std=dcfg.algo.config.init_noise_std,
        ).to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.delta_actor.load_state_dict(sd["actor_model_state_dict"])
        self.delta_actor.eval()
        for p in self.delta_actor.parameters():
            p.requires_grad = False

        self.delta_obs_keys = sorted(dcfg.obs.obs_dict.actor_obs)
        self.delta_obs_scales = dcfg.obs.obs_scales
        self.delta_ankle_only = bool(dcfg.env.config.get("anklePR", False))

        logger.info(f"frozen delta model loaded from {ckpt_path}")
        logger.info(f"  obs (sorted): {self.delta_obs_keys}")
        logger.info(f"  scope: {'4-DoF ankles only' if self.delta_ankle_only else 'all 23 DoF'}")

    # ----------------------------------------------------------------- delta obs
    def _build_delta_obs(self, policy_actions):
        """Rebuild the delta model's training-time observation for this step."""
        parts = {}
        for key in self.delta_obs_keys:
            if key == "actions_open_loop":
                # the action being corrected -- the policy's, now
                raw = policy_actions
            elif key == "actions":
                # the delta's own previous output, as during its training
                raw = self.last_delta_action
            else:
                raw = getattr(self, f"_get_obs_{key}")()
            parts[key] = raw * self.delta_obs_scales[key]
        # sorted order, matching _post_config_observation_callback
        return torch.cat([parts[k] for k in self.delta_obs_keys], dim=-1)

    @torch.no_grad()
    def _query_delta(self, policy_actions):
        if self.delta_actor is None:
            return torch.zeros_like(policy_actions)
        delta = self.delta_actor.act_inference(self._build_delta_obs(policy_actions))
        if self.delta_ankle_only:
            # The ankle-only model was trained with every other dim forced to zero
            # (delta_a_open_loop applies the mask AFTER the network). Reapply it, or
            # the untrained dims leak whatever the head happens to emit.
            mask = torch.zeros_like(delta)
            mask[:, ANKLE_IDX] = 1.0
            delta = delta * mask
        return delta

    # ------------------------------------------------------------------ torques
    def _compute_torques(self, actions):
        """The delta is added to the policy's action, then everything downstream --
        control type, RFI torque randomisation, torque clipping -- is the parent's
        unchanged. Delegating rather than reimplementing keeps this correct if the
        base controller changes."""
        delta = self._query_delta(actions)
        self.last_delta_action = delta.clone()
        return super()._compute_torques(actions + delta)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        if getattr(self, "last_delta_action", None) is not None:
            self.last_delta_action[env_ids] = 0.0
