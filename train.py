"""
TREINO — este é o arquivo que "liga" a IA.

O que ele faz:
  1. Carrega a calibração (gd_config.json).
  2. Dá um aviso de 5s pra você clicar na janela do jogo e deixar em foco.
  3. Começa a jogar sozinha, morrendo, reiniciando e aprendendo.
  4. A cada tentativa (episódio), mostra no terminal: número da tentativa,
     % máxima alcançada, recompensa total, epsilon (o quanto ela ainda
     está "explorando" vs "confiando no que aprendeu").
  5. Salva o cérebro dela periodicamente em checkpoints/gd_agent.pt —
     então se você fechar e rodar de novo, ela continua aprendendo do
     ponto onde parou, na mesma fase (ou em outra, se você trocar).

Pare com Ctrl+C a qualquer momento — o progresso já salvo não se perde.
"""

import os
import time
import csv

from config import load_config
from env import GDEnv
from agent import DQNAgent


def main():
    cfg = load_config()

    if cfg["game_region"] is None:
        print("Você ainda não calibrou a área do jogo.")
        print("Rode primeiro: python calibrate.py")
        return

    print("=" * 60)
    print("GD-AI — Treino iniciando em 5 segundos.")
    print("Clique na janela do Geometry Dash AGORA pra ela ficar em foco!")
    print("=" * 60)
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    env = GDEnv(cfg)
    agent = DQNAgent(
        n_actions=2,
        frame_stack=cfg["frame_stack"],
        checkpoint_path=cfg["checkpoint_path"],
    )

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    log_is_new = not os.path.exists(cfg["log_path"])
    log_file = open(cfg["log_path"], "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["episode", "reward_total", "best_percent", "epsilon", "steps_done"])

    episode = 0
    target_update_every = 2000  # em passos de treino
    save_every_episodes = 5

    try:
        while True:
            episode += 1
            state = env.reset()
            ep_reward = 0.0
            best_percent = 0
            done = False

            while not done:
                action = agent.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)

                agent.push_transition(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                best_percent = max(best_percent, info.get("percent", 0) or 0)

                loss = agent.train_step(batch_size=32)
                if loss is not None and agent.steps_done % target_update_every == 0:
                    agent.update_target()

            eps = agent.epsilon()
            print(f"[Tentativa {episode:5d}] recompensa={ep_reward:7.2f}  "
                  f"melhor%={best_percent:3d}  epsilon={eps:.3f}  "
                  f"steps_treino={agent.steps_done}")

            log_writer.writerow([episode, round(ep_reward, 3), best_percent, round(eps, 4), agent.steps_done])
            log_file.flush()

            if episode % save_every_episodes == 0:
                agent.save()
                print(f"  -> progresso salvo em {cfg['checkpoint_path']}")

    except KeyboardInterrupt:
        print("\nParando e salvando progresso...")
        agent.save()
        print("Salvo! Pode rodar 'python train.py' de novo depois pra continuar.")
    finally:
        env.close()
        log_file.close()


if __name__ == "__main__":
    main()
