import argparse
import json
import logging
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
import requests
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter
from racecar_utils import process_obs

LOGGER = logging.getLogger(__name__)


class SB3Agent:
    """Wrap a trained SB3 SAC policy for inference inside the HTTP client."""

    def __init__(
        self,
        model_path: str,
        frame_stack: int,
        deterministic: bool,
        device: str,
    ):
        self.frame_stack = frame_stack
        self.deterministic = deterministic
        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)
        LOGGER.info("Loading SAC policy from %s on %s", model_path, device)
        self.model = SAC.load(model_path, device=device)

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        processed = process_obs(obs)
        return processed[0]

    def _stack_frames(self) -> np.ndarray:
        frames = list(self.frames)
        while len(frames) < self.frame_stack:
            frames.append(frames[-1])
        stacked = np.stack(frames, axis=0)
        return np.expand_dims(stacked, axis=0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        frame = self._preprocess_obs(observation)
        if not self.frames:
            for _ in range(self.frame_stack):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
        stacked_obs = self._stack_frames()
        action, _ = self.model.predict(stacked_obs,
                                       deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)[0]


def connect(
    agent: SB3Agent,
    url: str = 'http://localhost:5000',
    timeout: int = 10,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Stream observations from the server and act with the loaded policy."""
    session = requests.Session()
    step_idx = 0

    while True:
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.error('GET request failed: %s', exc)
            break
        payload = json.loads(response.text)

        if payload.get('error'):
            LOGGER.error('Server error on GET: %s', payload['error'])
            break

        if payload.get('terminal'):
            LOGGER.info('Server signaled terminal before action loop.')
            break

        obs = np.array(payload['observation']).astype(np.uint8)
        if step_idx == 0:
            LOGGER.info('Initial observation shape: %s', obs.shape)

        step_idx += 1
        step_start = time.perf_counter()
        action_to_take = agent.act(obs)

        LOGGER.info('Step %d | action=%s', step_idx,
                    np.array2string(action_to_take, precision=3))

        if writer is not None:
            for i, val in enumerate(np.ravel(action_to_take)):
                writer.add_scalar(f'client/action_{i}', float(val), step_idx)

        try:
            response = session.post(url,
                                    json={'action': action_to_take.tolist()},
                                    timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.error('POST request failed: %s', exc)
            break
        payload = json.loads(response.text)

        if payload.get('error'):
            LOGGER.error('Server error on POST: %s', payload['error'])
            break

        elapsed = time.perf_counter() - step_start
        LOGGER.info('Step %d | terminal=%s | latency=%.3fs', step_idx,
                    payload.get('terminal', False), elapsed)

        if writer is not None:
            writer.add_scalar('client/latency_s', elapsed, step_idx)
            writer.add_scalar('client/terminal_flag',
                              int(bool(payload.get('terminal', False))),
                              step_idx)

        if payload.get('terminal'):
            LOGGER.info('Episode finished after %d steps.', step_idx)
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',
                        type=str,
                        default='http://localhost:5000',
                        help='The URL of the racecar server.')
    parser.add_argument('--model_path',
                        type=str,
                        default='runs/models_sac_racecar/best_model.zip',
                        help='Path to the trained SB3 model to load.')
    parser.add_argument('--frame_stack',
                        type=int,
                        default=6,
                        help='Number of grayscale frames to stack.')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Inference device for SB3 (e.g. cpu, cuda).')
    parser.add_argument('--deterministic',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Use deterministic actions (default: True).')
    parser.add_argument('--log_level',
                        type=str,
                        default='INFO',
                        help='Python logging level.')
    parser.add_argument(
        '--timeout',
        type=int,
        default=10,
        help='HTTP timeout in seconds for client-server calls.')
    parser.add_argument('--tb_logdir',
                        type=str,
                        default='runs/client',
                        help='Directory for TensorBoard logs.')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO))

    sac_agent = SB3Agent(model_path=args.model_path,
                         frame_stack=args.frame_stack,
                         deterministic=args.deterministic,
                         device=args.device)

    tb_writer = SummaryWriter(args.tb_logdir) if args.tb_logdir else None
    try:
        connect(sac_agent,
                url=args.url,
                timeout=args.timeout,
                writer=tb_writer)
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
