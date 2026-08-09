"""
Lê o texto de porcentagem da tela (ex: "45%") usando OCR (Tesseract),
pra servir de sinal de recompensa e detectar morte/vitória.

Requer o Tesseract instalado no sistema (ver README).
Se o Tesseract não estiver disponível, o sistema cai automaticamente
num modo alternativo baseado em diferença de pixels (menos preciso,
mas funciona sem instalar nada extra).
"""

import re
import numpy as np
import cv2

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False


class PercentReader:
    def __init__(self, sct, region):
        """
        sct: instância mss.mss()
        region: [x1, y1, x2, y2] absolutos da tela, onde fica o "NN%"
        """
        self.sct = sct
        x1, y1, x2, y2 = region
        self.monitor = {
            "left": int(x1),
            "top": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        }
        self.last_percent = 0
        self.available = TESSERACT_OK
        if not TESSERACT_OK:
            print("[aviso] pytesseract/tesseract não encontrado — "
                  "leitura de porcentagem desativada. Usando apenas "
                  "recompensa por tempo sobrevivido.")

    def read_percent(self):
        """Retorna um inteiro 0-100, ou None se não conseguir ler."""
        if not self.available:
            return None
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot)[:, :, :3]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # binariza para facilitar leitura de texto branco sobre fundo de jogo
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            scaled, config="--psm 7 -c tessedit_char_whitelist=0123456789%"
        )
        match = re.search(r"(\d{1,3})", text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                self.last_percent = val
                return val
        return None
