"""
Train SAC on racecar_gym for RL final project.

- Obs: 3 x 128 x 128 image (bird's-eye rgb_array)
- Action: (motor, steering) both in [-1, 1]
- Uses:
    - CnnPolicy (automatic CNN encoder)
    - VecFrameStack for frame stacking
    - RewardShapingWrapper for simple shaping
    - Checkpoint + Eval callbacks

Run:
    python train_sac_racecar.py
"""

import os
import cv2
from collections import deque
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from racecar_gym.env import RaceEnv

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# ------------------------
#  Env wrapper
# ------------------------


class RaceCarWrapper(gym.Env):
    """
    Wrap racecar_gym RaceEnv to:
      - enforce obs dtype/shape
      - add simple reward shaping

    Observation: 3 x 128 x 128 (float32 in [0, 1])
    Action: Box([-1,-1], [1,1])
    """

    metadata = {"render_modes": ["rgb_array_birds_eye"]}

    def __init__(
        self,
        scenario: str,
        reset_when_collision: bool,
        max_episode_steps: int | None = None,
        is_random_start: bool = True,
        is_eval: bool = False,
    ):
        super().__init__()

        self._env = RaceEnv(
            scenario=scenario,
            render_mode="rgb_array_birds_eye",
            reset_when_collision=reset_when_collision,
        )

        # expose action/obs spaces directly
        self.action_space = self._env.action_space

        # RaceEnv already returns (C,H,W) = (3, 96, 96) per spec
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 96, 96),
            dtype=np.float32,
        )

        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._is_random_start = is_random_start
        self._is_eval = is_eval

        # will use last action for shaping
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        RaceEnv returns uint8 0-255; convert to float32 0-1.
        """
        obs = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (96, 96))
        obs = obs.astype(np.float32) / 255.0
        obs = np.expand_dims(obs, axis=0)  # (1,96,96)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._is_random_start:
            if options is not None:
                options['mode'] = 'random'
            else:
                options = {'mode': 'random'}
        self._step_count = 0
        self._last_action[:] = 0.0
        obs, info = self._env.reset(seed=seed, options=options)
        obs = self._process_obs(obs)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Simple reward shaping:

            base_r = env reward (progress difference)
            shaped_r = base_r
                        + 0.5 * max(base_r, 0)         (encourage forward progress)
                        - 0.01 * |steering|            (discourage over-steer)

        This is intentionally simple & robust:只用 env reward + action，不依賴 info 裡的奇怪欄位。
        之後你可以自己改係數或加入速度等資訊。
        """
        self._step_count += 1

        obs, base_r, terminated, truncated, info = self._env.step(action)

        # reward shaping
        if self._is_eval:
            # no shaping during eval
            reward = float(base_r)
        else:
            progress_bonus = max(0.0, float(base_r))
            steering_penalty = 0.01 * float(abs(action[1]))
            reward = float(base_r) + 0.5 * progress_bonus - steering_penalty

        # optional time limit truncation
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            truncated = True

        obs = self._process_obs(obs)
        self._last_action = np.array(action, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_env_fn(
    scenario: str,
    reset_when_collision: bool,
    max_episode_steps: int | None = None,
    is_random_start: bool = True,
    is_eval: bool = False,
    seed: int = 0,
):
    """
    Factory for DummyVecEnv.
    """

    def _init():
        env = RaceCarWrapper(
            scenario=scenario,
            reset_when_collision=reset_when_collision,
            max_episode_steps=max_episode_steps,
            is_random_start=is_random_start,
            is_eval=is_eval,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return _init


# ------------------------
#  Training setup
# ------------------------


def argparse():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SAC on racecar_gym RaceEnv")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs/logs_sac_racecar",
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./runs/models_sac_racecar",
        help="Directory to save models.",
    )
    parser.add_argument("--frame_stack",
                        type=int,
                        default=6,
                        help="Number of frames to stack.")
    parser.add_argument("--num_envs",
                        type=int,
                        default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--lr",
                        type=float,
                        default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--total_timesteps",
                        type=int,
                        default=1_000_000,
                        help="Total timesteps to train.")
    parser.add_argument("--buffer_size",
                        type=int,
                        default=20000,
                        help="Replay buffer size.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="Batch size.")
    parser.add_argument("--train_freq",
                        type=int,
                        default=64,
                        help="Training frequency.")
    parser.add_argument("--gradient_steps",
                        type=int,
                        default=64,
                        help="Number of gradient steps.")
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help="Discount factor.")
    parser.add_argument("--tau",
                        type=float,
                        default=0.005,
                        help="Target network update rate.")

    parser.add_argument(
        "--scenario",
        type=str,
        default="circle_cw_competition_collisionStop",
        choices=[
            "circle_cw_competition_collisionStop",
            "austria_competition_collisionStop",
        ],
        help="RaceEnv scenario to train on.",
    )
    parser.add_argument(
        "--episode_seconds",
        type=float,
        default=None,
        help="Max env seconds per episode (default depends on scenario).",
    )
    return parser.parse_args()


def main():
    # ==== paths ====
    args = argparse()
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    # ==== env config ====
    # default episode length: 25s for circle, 100s for Austria (env spec)
    # 0.02 sec per frame  ->  steps = seconds / 0.02 :contentReference[oaicite:1]{index=1}
    if args.episode_seconds is None:
        if "circle" in args.scenario:
            episode_seconds = 25.0
        else:  # austria
            episode_seconds = 100.0
    else:
        episode_seconds = args.episode_seconds

    max_episode_steps = int(episode_seconds / 0.02)

    # for *_competition_collisionStop we want reset_when_collision = False
    # (for plain austria_competition/circle_cw you must keep True) :contentReference[oaicite:2]{index=2}
    reset_when_collision = not args.scenario.endswith(
        "_competition_collisionStop")

    num_envs = args.num_envs
    frame_stack = args.frame_stack

    # ==== make vec env ====
    env_fns = [
        make_env_fn(
            scenario=args.scenario,
            reset_when_collision=reset_when_collision,
            max_episode_steps=max_episode_steps,
            is_random_start=True,
            seed=100 + i,
        ) for i in range(num_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)  # log ep reward/len
    vec_env = VecFrameStack(vec_env,
                            n_stack=frame_stack,
                            channels_order="first")
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,  # reward 用原本的進度即可
        clip_obs=10.0,
    )

    # ==== eval env (single) ====
    eval_env_fn = make_env_fn(
        scenario=args.scenario,
        reset_when_collision=reset_when_collision,
        max_episode_steps=max_episode_steps,
        is_random_start=False,
        is_eval=True,
        seed=999,
    )
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env,
                             n_stack=frame_stack,
                             channels_order="first")
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    eval_env.training = False  # eval no stats update
    eval_env.obs_rms = vec_env.obs_rms  # share running stats
    eval_env.ret_rms = vec_env.ret_rms

    # ==== logger ====
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # ==== model ====
    model = SAC(
        policy="CnnPolicy",
        env=vec_env,
        policy_kwargs=dict(
            normalize_images=False,  # 這行是關鍵
        ),
        verbose=1,
        tensorboard_log=log_dir,
        # 這些是 reasonable default，你之後可以微調
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        target_entropy="auto",
    )
    model.set_logger(new_logger)

    # ==== callbacks ====
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // num_envs,  # 每 5 萬 env steps 存一次
        save_path=models_dir,
        name_prefix="sac_racecar",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=25_000 // num_envs,
        n_eval_episodes=5,
        deterministic=False,
        # SAC is stochastic; for eval keep False or True both OK
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # save final
    model_path = os.path.join(models_dir, "sac_racecar_final")
    model.save(model_path)
    print(f"Saved final model to {model_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
