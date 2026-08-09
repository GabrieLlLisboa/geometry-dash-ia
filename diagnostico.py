"""
DIAGNÓSTICO — rode isso pra CONFERIR se a leitura da porcentagem está
funcionando direito, antes de treinar.

Como usar:
1. Rode: python diagnostico.py
2. Jogue a fase manualmente (com o teclado, você mesmo).
3. Observe no terminal se o número mostrado bate com o "%" que aparece
   na tela do jogo, em tempo real.

Se os números baterem certinho -> calibração e OCR estão bons, pode
rodar o treino tranquilo.

Se os números ficarem pulando estranho, mostrando "None" toda hora, ou
totalmente errados -> a região de porcentagem foi calibrada errada, ou
o Tesseract não está instalado/no PATH. Rode calibrate.py de novo com
mais cuidado, ou revise a instalação do Tesseract (ver README.md).
"""

import time
import mss

from config import load_config
from percent_reader import PercentReader


def main():
    cfg = load_config()
    if cfg.get("percent_region") is None:
        print("Região de porcentagem não calibrada. Rode: python calibrate.py")
        return

    sct = mss.mss()
    reader = PercentReader(sct, cfg["percent_region"])

    if not reader.available:
        print("Tesseract não encontrado — instale e coloque no PATH (ver README).")
        return

    print("Lendo porcentagem a cada 0.3s. Jogue a fase manualmente e compare")
    print("com o número mostrado abaixo. Ctrl+C pra parar.\n")

    try:
        while True:
            val = reader.read_percent()
            print(f"leitura OCR: {val}", end="   \r")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nFim do diagnóstico.")


if __name__ == "__main__":
    main()
