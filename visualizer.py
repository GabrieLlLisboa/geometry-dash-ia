"""
VISUALIZADOR — janela separada que mostra em tempo real:

  1. O frame que a IA está enxergando (o que a câmera dela captura).
  2. Mapa de calor de SALIÊNCIA sobreposto: vermelho/amarelo = pixels que
     mais pesaram na decisão dela naquele instante (isso é literalmente
     "o que ela está olhando" pra decidir pular ou não).
  3. Barras de confiança (Q-values) de cada ação, com a escolhida
     destacada.
  4. Estatísticas ao vivo: tentativa atual, melhor %, epsilon (o quanto
     ela ainda está "no chute"), recompensa da tentativa atual.
  5. Gráfico de linha com o histórico de recompensa das últimas
     tentativas — pra você ver se a tendência é de melhora.

Isso roda numa janela do OpenCV (fora da janela do jogo). Pra fechar só
essa janela de visualização (o treino continua rodando), clique nela e
aperte "V". Aperte "Q" ali pra fechar de vez (também para o treino).
"""

import cv2
import numpy as np
import collections

WINDOW_NAME = "GD-AI - o que ela esta pensando"


class Visualizer:
    def __init__(self, scale=6, history_len=100):
        self.scale = scale
        self.reward_history = collections.deque(maxlen=history_len)
        self.enabled = True
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def _draw_heatmap_frame(self, frame01, saliency01):
        """frame01: (H,W) 0-1 float | saliency01: (H,W) 0-1 float ou None"""
        gray = (frame01 * 255).astype(np.uint8)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if saliency01 is not None:
            heat = (saliency01 * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            base = cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)

        h, w = base.shape[:2]
        base = cv2.resize(base, (w * self.scale, h * self.scale),
                           interpolation=cv2.INTER_NEAREST)
        return base

    def _draw_q_bars(self, canvas, q_values, chosen_action, x, y, w, h):
        labels = ["NAO PULAR", "PULAR"]
        qmin, qmax = float(np.min(q_values)), float(np.max(q_values))
        span = max(qmax - qmin, 1e-3)
        bar_h = h // len(q_values) - 8

        cv2.putText(canvas, "Confianca (Q-values):", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for i, q in enumerate(q_values):
            by = y + i * (bar_h + 8)
            norm = (q - qmin) / span
            bar_w = int(norm * (w - 140))
            color = (0, 220, 0) if i == chosen_action else (90, 90, 90)
            cv2.rectangle(canvas, (x, by), (x + max(bar_w, 2), by + bar_h), color, -1)
            cv2.rectangle(canvas, (x, by), (x + w - 140, by + bar_h), (120, 120, 120), 1)
            txt = f"{labels[i]}: {q:+.2f}"
            cv2.putText(canvas, txt, (x + w - 130, by + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_reward_chart(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), 1)
        cv2.putText(canvas, "Historico de recompensa por tentativa:", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if len(self.reward_history) < 2:
            return
        vals = list(self.reward_history)
        vmin, vmax = min(vals), max(vals)
        span = max(vmax - vmin, 1e-3)
        n = len(vals)
        pts = []
        for i, v in enumerate(vals):
            px = x + int(i / (n - 1) * (w - 1))
            py = y + h - int((v - vmin) / span * (h - 1))
            pts.append((px, py))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p1, p2, (0, 200, 255), 2)
        cv2.putText(canvas, f"min {vmin:.1f}", (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"max {vmax:.1f}", (x + w - 90, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def end_episode(self, total_reward):
        self.reward_history.append(total_reward)

    def render(self, frame01, saliency01, q_values, chosen_action,
               episode, best_percent, epsilon, ep_reward):
        if not self.enabled:
            return True

        frame_img = self._draw_heatmap_frame(frame01, saliency01)
        fh, fw = frame_img.shape[:2]

        panel_w = 380
        canvas_h = max(fh, 420)
        canvas = np.zeros((canvas_h, fw + panel_w, 3), dtype=np.uint8)
        canvas[:fh, :fw] = frame_img

        px = fw + 20
        cv2.putText(canvas, f"Tentativa: {episode}", (px, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Melhor %: {best_percent}", (px, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Epsilon (chute): {epsilon:.2f}", (px, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Recompensa atual: {ep_reward:.2f}", (px, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        action_txt = "PULANDO" if chosen_action == 1 else "PARADO"
        action_color = (0, 220, 0) if chosen_action == 1 else (150, 150, 150)
        cv2.putText(canvas, f"Acao: {action_txt}", (px, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, action_color, 2, cv2.LINE_AA)

        self._draw_q_bars(canvas, q_values, chosen_action, px, 210, panel_w - 40, 90)
        self._draw_reward_chart(canvas, px, 340, panel_w - 40, 60)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            self.enabled = False
            cv2.destroyWindow(WINDOW_NAME)
        elif key == ord('q'):
            return False  # sinaliza pra parar o treino também
        return True

    def close(self):
        cv2.destroyAllWindows()
