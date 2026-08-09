"""
Controle de teclado. Usa pydirectinput no Windows (mais confiável pra
jogos que leem input em baixo nível) e cai pra pyautogui em outros SOs.
"""

import platform

if platform.system() == "Windows":
    import pydirectinput as backend
    backend.PAUSE = 0.0
else:
    import pyautogui as backend
    backend.PAUSE = 0.0


class Controller:
    def __init__(self, jump_key="space"):
        self.jump_key = jump_key
        self.holding = False

    def press_jump(self):
        """Aperta e segura o pulo."""
        if not self.holding:
            backend.keyDown(self.jump_key)
            self.holding = True

    def release_jump(self):
        """Solta o pulo."""
        if self.holding:
            backend.keyUp(self.jump_key)
            self.holding = False

    def tap_restart(self):
        """Aperta espaço rapidamente pra reiniciar a fase após morrer."""
        self.release_jump()
        backend.press(self.jump_key)

    def do_action(self, action):
        """action: 0 = não fazer nada / soltar, 1 = pular / segurar."""
        if action == 1:
            self.press_jump()
        else:
            self.release_jump()
