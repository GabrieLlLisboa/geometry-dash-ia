"""
Captura de tela da área do jogo, usando mss (rápido, funciona em
Windows/Linux/Mac).
"""

import numpy as np
import mss
import cv2


class ScreenCapturer:
    def __init__(self, region):
        """
        region: [x1, y1, x2, y2] em coordenadas absolutas de tela.
        """
        x1, y1, x2, y2 = region
        self.monitor = {
            "left": int(x1),
            "top": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        }
        self.sct = mss.mss()

    def grab_raw(self):
        """Retorna o frame cru em BGR (numpy array HxWx3)."""
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot)[:, :, :3]  # remove canal alpha
        return frame

    def grab_processed(self, size=(84, 84)):
        """Retorna o frame em escala de cinza, redimensionado, normalizado [0,1]."""
        frame = self.grab_raw()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def close(self):
        self.sct.close()
