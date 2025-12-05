import os
import numpy as np
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import sync_envs_normalization

class LastReplayBufferCallback(BaseCallback):
    def __init__(self, filename="replay_buffer.pkl", save_freq=100000):
        super().__init__()
        self.filename = filename
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(self.filename)
        return True


class CustomEvalCallback(BaseCallback):
    """
    Mimic SB3 EvalCallback, but also log base_r (info['reward']):

    - 每 eval_freq 步，在 eval_env 上跑 n_eval_episodes
    - 記錄:
        eval/mean_reward        : 使用 env.step 回傳的 reward（你的 shaping reward）
        eval/base_mean_reward   : 用 info['reward'] 累積的原始 env reward
        eval/mean_ep_length
    - 若 mean_reward 變好，存 best model
    - 可以選擇寫 evaluations.npz
    """

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        warn: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.best_mean_reward = -np.inf
        self.evaluations_timesteps: List[int] = []
        self.evaluations_results: List[List[float]] = []
        self.evaluations_length: List[List[int]] = []
        self.evaluations_base: List[List[float]] = []  # base_r per episode

    def _init_callback(self) -> None:
        # 確保 eval_env 是 VecEnv；你本來就是 DummyVecEnv，所以基本上 OK
        assert isinstance(self.eval_env, VecEnv), "eval_env 必須是 VecEnv"

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _make_step_callback(self, base_returns_per_env: np.ndarray,
                            episode_base_returns: List[float]):
        """
        建一個傳給 evaluate_policy 的 step callback，用來在每一步累積 base_r。
        base_returns_per_env: shape = (n_envs,)
        episode_base_returns: 存每個 episode 完成時的 base 累積
        """

        def _step_callback(locals_: Dict[str, Any], globals_: Dict[str, Any]):
            # 這些 key 是 evaluate_policy locals() 裡會有的
            i = locals_["i"]  # 第幾個 env
            info = locals_["info"]
            done = locals_["done"]

            base_r = float(info.get("reward", 0.0))
            base_returns_per_env[i] += base_r

            if done:
                # 該 env 的一個 episode 結束 → 把 base 累積存起來，並 reset
                episode_base_returns.append(base_returns_per_env[i])
                base_returns_per_env[i] = 0.0

        return _step_callback

    def _on_step(self) -> bool:
        # 每 eval_freq steps 做一次 evaluation
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True

        # 如果有 VecNormalize，先把 training 和 eval 的 running stats 同步
        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_env)

        # 用來記錄 base_r
        n_envs = self.eval_env.num_envs
        base_returns_per_env = np.zeros(n_envs, dtype=np.float32)
        episode_base_returns: List[float] = []

        # 建立 step callback（在 eval loop 中每一步被呼叫）
        step_cb = self._make_step_callback(
            base_returns_per_env,
            episode_base_returns,
        )

        # 評估 policy，回傳 per-episode shaped rewards / lengths
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            return_episode_rewards=True,
            warn=self.warn,
            callback=step_cb,
        )

        # 這裡 episode_rewards 是 shaping reward 的總和
        episode_rewards = list(episode_rewards)
        episode_lengths = list(episode_lengths)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_ep_length = float(np.mean(episode_lengths))
        std_ep_length = float(np.std(episode_lengths))

        # base_r 的統計
        mean_base = float(np.mean(episode_base_returns)) if len(
            episode_base_returns) > 0 else 0.0
        std_base = float(np.std(episode_base_returns)) if len(
            episode_base_returns) > 0 else 0.0

        if self.verbose >= 1:
            print(f"[Eval] num_timesteps={self.num_timesteps}, "
                  f"reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"base={mean_base:.2f} +/- {std_base:.2f}, "
                  f"len={mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        # 寫入 evaluations.npz（如果有 log_path）
        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            self.evaluations_base.append(episode_base_returns)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                base_rewards=self.evaluations_base,
            )

        # TensorBoard logging
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/base_mean_reward", mean_base)
        self.logger.record("eval/base_std_reward", std_base)
        self.logger.record("time/total_timesteps",
                           self.num_timesteps,
                           exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        # best model 判斷邏輯：用 shaping reward 的 mean_reward
        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward, saving model...")
            self.best_mean_reward = mean_reward
            if self.best_model_save_path is not None:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                save_path = os.path.join(self.best_model_save_path,
                                         "best_model")
                self.model.save(save_path)

        return True
