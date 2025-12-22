"""
Train SAC on racecar_gym for RL final project.

Observation: 1 x 96 x 96 (uint8 in [0, 255])
Action: Box([-1,-1], [1,1])
"""

import os
from typing import List

import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecFrameStack,
)
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from EnvWrapper import MultiScenarioWrapper, RaceCarWrapper
from Callback import CustomEvalCallback, LastReplayBufferCallback
from racecar_models import RaceCarMiniVGG

DEFAULT_MIXED_SCENARIOS = [
    "circle_cw_competition_collisionStop",
    "austria_competition_collisionStop",
]


def parse_probabilities(prob_str: str | None,
                        num_items: int) -> List[float] | None:
    """Parse comma-separated probabilities; returns None if not provided."""
    if prob_str is None:
        return None
    parts = [p.strip() for p in prob_str.split(",") if p.strip()]
    if len(parts) != num_items:
        raise ValueError(
            f"Expected {num_items} probabilities, got {len(parts)}.")
    return [float(p) for p in parts]


def scenario_seconds(scenario: str, override_seconds: float | None) -> float:
    """Return default episode seconds per scenario unless overridden."""
    if override_seconds is not None:
        return override_seconds
    return 25.0 if "circle" in scenario else 100.0


def make_env_fn(
    scenarios: list[str],
    scenario_probs: list[float] | None,
    reset_when_collision: bool,
    max_eval_episode_steps_by_scenario: dict[str, int],
    is_random_start: bool = True,
    is_eval: bool = False,
    seed: int = 0,
):
    """
    Factory for DummyVecEnv supporting single or mixed scenarios.
    """

    def _init():
        if len(scenarios) == 1:
            scenario = scenarios[0]
            env = RaceCarWrapper(
                scenario=scenario,
                reset_when_collision=reset_when_collision,
                max_eval_episode_steps=max_eval_episode_steps_by_scenario.get(
                    scenario),
                is_random_start=is_random_start,
                is_eval=is_eval,
            )
        else:
            env = MultiScenarioWrapper(
                scenarios=scenarios,
                scenario_probs=scenario_probs,
                reset_when_collision=reset_when_collision,
                max_eval_episode_steps_by_scenario=
                max_eval_episode_steps_by_scenario,
                is_random_start=is_random_start,
                is_eval=is_eval,
                seed=seed,
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
        "--encoder",
        type=str,
        default="sb3_cnn",
        choices=["sb3_cnn", "racecar_minivgg"],
        help="Pixel encoder to use inside SAC policy.",
    )
    parser.add_argument("--encoder_features_dim",
                        type=int,
                        default=256,
                        help="Latent dim for the lightweight encoder.")
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
        "--use_mixed_scenarios",
        action="store_true",
        help="Sample both circle and austria scenarios on reset.",
    )
    parser.add_argument(
        "--scenario_probs",
        type=str,
        default=None,
        help="Comma-separated sampling probabilities (only when mixing).",
    )
    parser.add_argument(
        "--eval_episode_seconds",
        type=float,
        default=None,
        help="Max eval env seconds per episode (default depends on scenario).",
    )
    parser.add_argument("--ckpt_model",
                        type=str,
                        default=None,
                        help="Path to checkpoint model to load.")
    parser.add_argument(
        "--ckpt_replay_buffer",
        type=str,
        default=None,
        help="Path to checkpoint replay buffer to load.",
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
    scenarios = (DEFAULT_MIXED_SCENARIOS
                 if args.use_mixed_scenarios else [args.scenario])
    scenario_probs = (parse_probabilities(args.scenario_probs, len(scenarios))
                      if args.use_mixed_scenarios else None)

    # default episode length: 25s for circle, 100s for Austria (env spec)
    # 0.02 sec per frame  ->  steps = seconds / 0.02
    max_eval_episode_steps_by_scenario = {
        scenario:
        int(scenario_seconds(scenario, args.eval_episode_seconds) / 0.02)
        for scenario in scenarios
    }

    # for *_competition_collisionStop we want reset_when_collision = False
    # (for plain austria_competition/circle_cw you must keep True)
    reset_when_collision = not scenarios[0].endswith(
        "_competition_collisionStop")
    for scenario in scenarios[1:]:
        if scenario.endswith("_competition_collisionStop") != scenarios[
                0].endswith("_competition_collisionStop"):
            raise ValueError(
                "Mixed scenarios with conflicting reset_when_collision flags are not supported."
            )

    num_envs = args.num_envs
    frame_stack = args.frame_stack
    policy_kwargs = dict(normalize_images=True)
    if args.encoder == "racecar_minivgg":
        policy_kwargs.update(
            features_extractor_class=RaceCarMiniVGG,
            features_extractor_kwargs=dict(
                features_dim=args.encoder_features_dim),
        )

    # ==== make vec env ====
    env_fns = [
        make_env_fn(
            scenarios=scenarios,
            scenario_probs=scenario_probs,
            reset_when_collision=reset_when_collision,
            max_eval_episode_steps_by_scenario=
            max_eval_episode_steps_by_scenario,
            is_random_start=True,
            seed=100 + i,
        ) for i in range(num_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)  # log ep reward/len
    vec_env = VecFrameStack(vec_env,
                            n_stack=frame_stack,
                            channels_order="first")

    # ==== eval env (single) ====
    eval_env_fn = make_env_fn(
        scenarios=scenarios,
        scenario_probs=scenario_probs,
        reset_when_collision=reset_when_collision,
        max_eval_episode_steps_by_scenario=max_eval_episode_steps_by_scenario,
        is_random_start=False,
        is_eval=True,
        seed=999,
    )
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env,
                             n_stack=frame_stack,
                             channels_order="first")

    # ==== logger ====
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # ==== model ====
    if args.ckpt_model is not None:
        print(f"Loading model from {args.ckpt_model} ...")
        model = SAC.load(args.ckpt_model, env=vec_env, print_system_info=True)
        if args.ckpt_replay_buffer is not None:
            print(f"Loading replay buffer from {args.ckpt_replay_buffer} ...")
            model.load_replay_buffer(args.ckpt_replay_buffer)
    else:
        model = SAC(
            policy="CnnPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            # 這些是 reasonable default，你之後可以微調
            learning_starts=10_000,
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
        save_replay_buffer=False,
    )

    last_replay_buffer_callback = LastReplayBufferCallback(
        save_freq=50_000 // num_envs,
        filename=os.path.join(models_dir, "sac_racecar_replay_buffer.pkl"),
    )

    eval_callback = CustomEvalCallback(
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
        callback=[
            checkpoint_callback, eval_callback, last_replay_buffer_callback
        ],
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
