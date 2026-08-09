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
import random

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
        epsilon_decay_steps=cfg.get("epsilon_decay_steps", 8000),
        epsilon_end=cfg.get("epsilon_end", 0.05),
        target_tau=cfg.get("target_tau", 0.01),
    )

    viz = None
    if cfg.get("show_visualizer", True):
        from visualizer import Visualizer
        viz = Visualizer(scale=cfg.get("visualizer_scale", 6))
        print("Janela 'GD-AI - o que ela esta pensando' aberta.")
        print("  Nela: aperte 'V' pra esconder so a visualizacao, 'Q' pra parar tudo.")

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    log_is_new = not os.path.exists(cfg["log_path"])
    log_file = open(cfg["log_path"], "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["episode", "reward_total", "best_percent", "epsilon", "steps_done"])

    episode = 0
    save_every_episodes = 5
    saliency_every_n = cfg.get("saliency_every_n_steps", 3)
    global_step = 0
    keep_running = True

    try:
        while keep_running:
            episode += 1
            state = env.reset()
            ep_reward = 0.0
            best_percent = 0
            done = False

            while not done:
                global_step += 1

                # a cada N passos calcula tambem o mapa de saliencia
                # (mais caro: usa gradiente) pra alimentar a visualizacao;
                # nos demais passos, so a escolha rapida de acao.
                if viz is not None and viz.enabled and global_step % saliency_every_n == 0:
                    action, q_values, saliency = agent.compute_saliency(state)
                    # ainda aplica epsilon-greedy por cima da escolha "pura"
                    eps = agent.epsilon()
                    if random.random() < eps:
                        action = random.randrange(agent.n_actions)
                else:
                    action = agent.select_action(state, explore=True)
                    q_values = agent.last_q_values
                    saliency = None

                next_state, reward, done, info = env.step(action)

                agent.push_transition(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                best_percent = max(best_percent, info.get("percent", 0) or 0)

                loss = agent.train_step(batch_size=32)

                if viz is not None:
                    keep_running = viz.render(
                        frame01=state[-1],
                        saliency01=saliency,
                        q_values=q_values,
                        chosen_action=action,
                        episode=episode,
                        best_percent=best_percent,
                        epsilon=agent.epsilon(),
                        ep_reward=ep_reward,
                    )
                    if not keep_running:
                        done = True

            if viz is not None:
                viz.end_episode(ep_reward)

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
        if viz is not None:
            viz.close()


if __name__ == "__main__":
    main()
