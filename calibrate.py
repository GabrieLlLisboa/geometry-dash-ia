"""
CALIBRAÇÃO — rode isso ANTES do treino, uma vez (ou sempre que mudar
a resolução/posição da janela do Geometry Dash).

Como funciona:
1. Abra o Geometry Dash e entre numa fase (pode ficar parado no início dela).
2. Rode: python calibrate.py
3. Siga as instruções no terminal: você vai posicionar o mouse em 2 pontos
   (canto superior-esquerdo e inferior-direito da ÁREA DE JOGO, sem HUD do
   Windows/barra de tarefas) e apertar ENTER pra marcar cada um.
4. Depois vai marcar a posição do texto de PORCENTAGEM (o "45%" que aparece
   no topo da tela durante a fase).

Tudo é salvo em gd_config.json.
"""

import time
import sys

try:
    import pyautogui
except ImportError:
    print("Faltando dependência. Rode: pip install -r requirements.txt")
    sys.exit(1)

from config import load_config, save_config


def wait_for_enter(msg):
    input(msg)
    pos = pyautogui.position()
    return pos.x, pos.y


def countdown_point(label, seconds=4):
    print(f"\n>> Posicione o mouse em: {label}")
    for i in range(seconds, 0, -1):
        print(f"   capturando em {i}...", end="\r")
        time.sleep(1)
    pos = pyautogui.position()
    print(f"   Ponto capturado: {pos}                  ")
    return pos.x, pos.y


def main():
    print("=" * 60)
    print("CALIBRAÇÃO DO GD-AI")
    print("=" * 60)
    print("Deixe o Geometry Dash aberto e visível, numa fase, parado")
    print("no início (não precisa estar jogando ainda).\n")

    input("Pressione ENTER para começar...")

    x1, y1 = countdown_point("o CANTO SUPERIOR ESQUERDO da área de jogo")
    x2, y2 = countdown_point("o CANTO INFERIOR DIREITO da área de jogo")

    print("\nAgora vamos marcar onde fica o texto de PORCENTAGEM (ex: 45%).")
    print("Normalmente fica no topo da tela, centralizado.")
    px1, py1 = countdown_point("o CANTO SUPERIOR ESQUERDO do texto de %")
    px2, py2 = countdown_point("o CANTO INFERIOR DIREITO do texto de %")

    cfg = load_config()
    cfg["game_region"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    cfg["percent_region"] = [min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)]
    save_config(cfg)

    print("\nCalibração salva em gd_config.json!")
    print("Região do jogo:", cfg["game_region"])
    print("Região da porcentagem:", cfg["percent_region"])
    print("\nAgora é só rodar: python train.py")


if __name__ == "__main__":
    main()
