"""Environment wrappers for racecar_gym scenarios."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
from racecar_gym.env import RaceEnv

from racecar_utils import process_obs


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
        max_eval_episode_steps: int | None = None,
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
            low=0,
            high=255,
            shape=(1, 96, 96),
            dtype=np.uint8,
        )

        self._max_eval_episode_steps = max_eval_episode_steps
        self._step_count = 0
        self._is_random_start = is_random_start
        self._is_eval = is_eval

        # will use last action for shaping
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        return process_obs(obs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._is_random_start:
            if options is not None:
                options["mode"] = "random"
            else:
                options = {"mode": "random"}
        self._step_count = 0
        self._last_action[:] = 0.0
        obs, info = self._env.reset(seed=seed, options=options)
        obs = self._process_obs(obs)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Reward shaping 版本（使用 info）：

            base_r = env reward (progress difference)

            shaped_r = base_r
                       + 0.5 * max(base_r, 0)       # 再補強一點前進進度
                       + 20.0 * speed               # 鼓勵有速度，不要原地打轉
                       - 0.01 * |steering|          # 緩和地懲罰過度轉向
                       - 1.0  if collision          # 撞牆 / 撞車 penalty
                       - 0.5  if wrong_way          # 反方向開 penalty
        """
        self._step_count += 1

        obs, base_r, terminated, truncated, info = self._env.step(action)

        # 從 info 抽出需要的欄位（用 get 避免沒有 key 時炸掉）
        wall_collision = bool(info.get("wall_collision", False))
        opponent_collisions = info.get("opponent_collisions", [])
        collided = wall_collision or (len(opponent_collisions) > 0)

        wrong_way = bool(info.get("wrong_way", False))
        velocity = np.array(info.get("velocity", np.zeros(6,
                                                          dtype=np.float32)),
                            dtype=np.float32)

        # 只取 x, y 兩個平面速度來算 speed（避免轉動角速度影響）
        planar_speed = float(np.linalg.norm(velocity[:2]))

        reward = float(base_r)

        # 1) 補強「正向進度」
        forward_progress = max(0.0, reward)
        reward += 0.5 * forward_progress

        # 2) 鼓勵有速度（scale 要和 base_r 同一個量級）
        # 你貼的樣本裡速度很小（1e-4 等級），所以乘上 20 大概是 1e-3
        reward += 20.0 * planar_speed

        # 3) 懲罰過度轉向，但不要過大（避免車不敢轉彎）
        steering_penalty = 0.01 * float(abs(action[1]))
        reward -= steering_penalty

        # 4) 碰撞 / 反向 penalty
        if collided:
            reward -= 1.0  # 撞牆 / 撞車，給一個明顯的負號 signal
        if wrong_way:
            reward -= 0.5  # 反方向開也扣分

        # optional time limit truncation
        if (self._is_eval and self._max_eval_episode_steps is not None
                and self._step_count >= self._max_eval_episode_steps):
            truncated = True

        obs = self._process_obs(obs)
        self._last_action = np.array(action, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class MultiScenarioWrapper(gym.Env):
    """
    Switch scenarios on reset and hard-reset the simulator by recreating
    RaceCarWrapper each episode.

    Adds `info["scenario"]` for logging the active track.
    """

    metadata = {"render_modes": ["rgb_array_birds_eye"]}

    def __init__(
        self,
        scenarios: Sequence[str],
        scenario_probs: Sequence[float] | None,
        reset_when_collision: bool,
        max_eval_episode_steps_by_scenario: dict[str, int | None],
        is_random_start: bool = True,
        is_eval: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        if len(scenarios) < 2:
            raise ValueError(
                "MultiScenarioWrapper expects at least two scenarios.")
        self._scenarios = list(scenarios)
        self._scenario_probs = self._normalize_probs(scenario_probs)
        self._reset_when_collision = reset_when_collision
        self._max_steps_by_scenario = max_eval_episode_steps_by_scenario
        self._is_random_start = is_random_start
        self._is_eval = is_eval
        self._rng = np.random.default_rng(seed)
        self._episode_seed = seed
        self._episode_idx = 0
        self._env: RaceCarWrapper | None = None
        self._current_scenario: str | None = None

        # bootstrap action/obs spaces using the first scenario
        probe_env = RaceCarWrapper(
            scenario=self._scenarios[0],
            reset_when_collision=self._reset_when_collision,
            max_eval_episode_steps=self._max_steps_by_scenario.get(
                self._scenarios[0]),
            is_random_start=self._is_random_start,
            is_eval=self._is_eval,
        )
        self.action_space = probe_env.action_space
        self.observation_space = probe_env.observation_space
        probe_env.close()

    def _normalize_probs(self, probs: Sequence[float] | None) -> np.ndarray:
        if probs is None:
            return np.full(len(self._scenarios), 1.0 / len(self._scenarios))
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape[0] != len(self._scenarios):
            raise ValueError(
                "scenario_probs length must match scenarios length.")
        total = probs.sum()
        return probs / total

    def _sample_scenario(self) -> str:
        idx = int(
            self._rng.choice(len(self._scenarios), p=self._scenario_probs))
        return self._scenarios[idx]

    def _hard_reset_env(
            self, *, seed: int | None,
            options: dict | None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._env is not None:
            self._env.close()
        self._current_scenario = self._sample_scenario()
        max_steps = self._max_steps_by_scenario.get(self._current_scenario)
        self._env = RaceCarWrapper(
            scenario=self._current_scenario,
            reset_when_collision=self._reset_when_collision,
            max_eval_episode_steps=max_steps,
            is_random_start=self._is_random_start,
            is_eval=self._is_eval,
        )
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info or {})
        info["scenario"] = self._current_scenario
        return obs, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._episode_idx += 1
        reset_seed = seed if seed is not None else self._episode_seed + self._episode_idx
        obs, info = self._hard_reset_env(seed=reset_seed, options=options)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info or {})
        info["scenario"] = self._current_scenario
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._env is None:
            return None
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
