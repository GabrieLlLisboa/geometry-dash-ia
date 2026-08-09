"""
Configurações do projeto GD-AI.
Depois de rodar calibrate.py, este arquivo é atualizado automaticamente
com a área da tela onde o jogo está.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gd_config.json")

DEFAULT_CONFIG = {
    # Região do jogo na tela: [x1, y1, x2, y2] (canto sup. esq. e inf. dir.)
    "game_region": None,

    # Região onde fica o texto de porcentagem (ex: "45%") em coordenadas
    # ABSOLUTAS de tela. É definida pelo calibrate.py.
    "percent_region": None,

    # Tecla usada para pular / segurar
    "jump_key": "space",

    # Tempo entre cada "passo" da IA (segundos). Menor = mais reativo,
    # porém mais pesado pra CPU/GPU.
    "step_delay": 0.033,  # ~30 passos por segundo

    # Tamanho da imagem que a rede neural enxerga
    "frame_size": [84, 84],

    # Quantos frames empilhados formam um "estado" (dá noção de movimento)
    "frame_stack": 4,

    # Onde salvar o cérebro da IA
    "checkpoint_path": os.path.join(os.path.dirname(__file__), "checkpoints", "gd_agent.pt"),

    # Onde salvar logs de treino (pra ver evolução)
    "log_path": os.path.join(os.path.dirname(__file__), "logs", "train_log.csv"),
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
