"""
GDEnv: embrulha o Geometry Dash real numa interface parecida com a do
Gymnasium (reset / step), pra podermos treinar um agente de RL nele.

Detecção de morte: como não lemos a memória do jogo, usamos DOIS sinais
combinados:
  1) Queda brusca da % lida por OCR (voltou pra perto de 0 sem ter
     chegado em 100) -> morreu.
  2) Se o OCR não estiver disponível, usamos uma heurística baseada em
     "quase nenhuma mudança de pixel por N frames seguidos" (tela de
     morte / menu parado) combinada com um teto de passos por tentativa.
"""

import time
import collections
import numpy as np

from capture import ScreenCapturer
from percent_reader import PercentReader
from controller import Controller


class GDEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        region = cfg["game_region"]
        if region is None:
            raise RuntimeError(
                "Área do jogo não calibrada. Rode: python calibrate.py"
            )

        self.capturer = ScreenCapturer(region)
        self.controller = Controller(cfg["jump_key"])

        self.percent_reader = None
        if cfg.get("percent_region"):
            self.percent_reader = PercentReader(self.capturer.sct, cfg["percent_region"])

        self.frame_size = tuple(cfg["frame_size"])
        self.stack_n = cfg["frame_stack"]
        self.step_delay = cfg["step_delay"]

        self.frames = collections.deque(maxlen=self.stack_n)

        self.max_steps_no_progress = 300  # trava de segurança (~10s parado)
        self._steps_since_progress = 0
        self._best_percent_this_try = 0
        self._prev_percent = 0
        self._episode_steps = 0
        self.max_episode_steps = 6000  # ~200s por tentativa, teto de segurança

    # ---------- API estilo Gym ----------

    def reset(self):
        self.controller.release_jump()
        # dá um tapzinho de espaço pra garantir que a fase (re)começa
        self.controller.tap_restart()
        time.sleep(0.5)

        self._best_percent_this_try = 0
        self._prev_percent = 0
        self._steps_since_progress = 0
        self._episode_steps = 0

        frame = self.capturer.grab_processed(self.frame_size)
        self.frames.clear()
        for _ in range(self.stack_n):
            self.frames.append(frame)
        return self._get_state()

    def step(self, action):
        self.controller.do_action(action)
        time.sleep(self.step_delay)

        frame = self.capturer.grab_processed(self.frame_size)
        self.frames.append(frame)
        self._episode_steps += 1

        reward, done, info = self._compute_reward_done(frame)

        if done:
            self.controller.release_jump()

        return self._get_state(), reward, done, info

    def close(self):
        self.controller.release_jump()
        self.capturer.close()

    # ---------- internos ----------

    def _get_state(self):
        return np.stack(self.frames, axis=0)  # shape: (stack_n, H, W)

    def _compute_reward_done(self, frame):
        info = {}
        percent = None
        if self.percent_reader is not None:
            percent = self.percent_reader.read_percent()

        if percent is not None:
            info["percent"] = percent

            # venceu a fase
            if percent >= 100:
                return 20.0, True, info

            # morreu: porcentagem caiu bastante em relação ao pico atingido
            if percent < self._best_percent_this_try - 5 and self._best_percent_this_try > 2:
                return -1.0, True, info

            # progresso: recompensa proporcional ao quanto avançou
            progress = max(0, percent - self._best_percent_this_try)
            if progress > 0:
                self._best_percent_this_try = percent
                self._steps_since_progress = 0
                reward = 0.1 + progress * 0.05
            else:
                self._steps_since_progress += 1
                reward = 0.01  # recompensinha por continuar vivo

            self._prev_percent = percent

            if self._steps_since_progress > self.max_steps_no_progress:
                return -0.5, True, info

            if self._episode_steps > self.max_episode_steps:
                return reward, True, info

            return reward, False, info

        # -------- modo sem OCR: heurística por tempo sobrevivido --------
        self._episode_steps += 1
        reward = 0.02
        done = self._episode_steps > self.max_episode_steps
        return reward, done, info
