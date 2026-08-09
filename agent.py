"""
Agente DQN (Deep Q-Network) — a "IA" propriamente dita.

Versão melhorada:
  - Arquitetura DUELING (separa "quão bom é esse estado" de "quão melhor
    é pular vs não pular aqui" — aprende mais rápido e mais estável).
  - DOUBLE DQN (usa a policy_net pra ESCOLHER a ação e a target_net pra
    AVALIAR ela — evita que a IA fique "otimista demais" com ações que
    na real são ruins, um problema clássico do DQN simples).
  - Soft update (Polyak averaging) do alvo a cada passo, em vez de copiar
    a rede inteira de vez em quando — deixa o aprendizado mais suave.
  - compute_saliency(): calcula, via gradiente, QUAIS PIXELS da imagem
    mais influenciaram a decisão da IA — é o que alimenta o mapa de calor
    da janela de visualização (visualizer.py).
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        # Ramo de VALOR: "quão bom é, no geral, esse momento da fase?"
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        # Ramo de VANTAGEM: "o quanto MELHOR é pular vs não pular, aqui?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        # combina os dois ramos: Q = V + (A - média(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s), np.array(a), np.array(r, dtype=np.float32),
            np.array(s2), np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions=2, frame_stack=4, checkpoint_path=None,
                 lr=1e-4, gamma=0.99, device=None,
                 epsilon_decay_steps=8000, epsilon_end=0.05,
                 target_tau=0.01):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.checkpoint_path = checkpoint_path
        self.target_tau = target_tau  # fração de "mistura" do soft update

        self.policy_net = DuelingQNetwork(frame_stack, n_actions).to(self.device)
        self.target_net = DuelingQNetwork(frame_stack, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # guarda o último Q-values calculado, pra visualização não
        # precisar recalcular toda hora
        self.last_q_values = np.zeros(n_actions, dtype=np.float32)

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.load(checkpoint_path)
                print(f"[agent] checkpoint carregado de {checkpoint_path} "
                      f"(steps_done={self.steps_done})")
            except RuntimeError as e:
                print("=" * 60)
                print("[aviso] o checkpoint salvo não é compatível com a")
                print("arquitetura atual da rede (provavelmente foi salvo")
                print("com uma versão diferente do código). Começando um")
                print("cérebro NOVO do zero. O arquivo antigo não foi")
                print("apagado, só ignorado — se quiser, delete-o manualmente:")
                print(f"  {checkpoint_path}")
                print("=" * 60)

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, explore=True):
        eps = self.epsilon() if explore else 0.0
        if explore and random.random() < eps:
            action = random.randrange(self.n_actions)
            # mesmo em ação aleatória, guarda os Q-values reais pra
            # visualização não mentir sobre o que a rede "achava"
            with torch.no_grad():
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                self.last_q_values = self.policy_net(s).cpu().numpy()[0]
            return action
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            self.last_q_values = q.cpu().numpy()[0]
            return int(q.argmax(dim=1).item())

    def compute_saliency(self, state, action=None):
        """
        Calcula qual região da imagem mais "pesou" na decisão, via
        gradiente da Q-value escolhida em relação aos pixels de entrada
        (vanilla saliency map, Simonyan et al. 2013).
        Retorna: (action_usada, q_values, mapa_saliencia HxW normalizado 0-1)
        """
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        s.requires_grad_(True)

        q = self.policy_net(s)
        if action is None:
            action = int(q.argmax(dim=1).item())

        self.policy_net.zero_grad(set_to_none=True)
        q[0, action].backward()

        grad = s.grad.detach().abs().squeeze(0).cpu().numpy()  # (stack, H, W)
        saliency = grad.max(axis=0)  # combina os canais empilhados
        smax = saliency.max()
        if smax > 1e-8:
            saliency = saliency / smax
        return action, q.detach().cpu().numpy()[0], saliency

    def push_transition(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self, batch_size=32):
        if len(self.buffer) < max(batch_size, 1000):
            return None

        s, a, r, s2, done = self.buffer.sample(batch_size)
        s = torch.from_numpy(s).float().to(self.device)
        s2 = torch.from_numpy(s2).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # DOUBLE DQN: a policy_net escolhe a melhor ação do próximo
            # estado, a target_net só avalia o quão boa ela é. Isso evita
            # a IA superestimar ações ruins (bug clássico do DQN simples).
            next_actions = self.policy_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            target = r + (1 - done) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self._soft_update_target()

        self.steps_done += 1
        return loss.item()

    def _soft_update_target(self):
        """Mistura devagarinho os pesos da target_net em direção à
        policy_net, em vez de copiar tudo de vez — deixa o "alvo" do
        treino mais estável (menos oscilação/instabilidade)."""
        tau = self.target_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(),
                                             self.policy_net.parameters()):
                target_param.mul_(1.0 - tau).add_(param, alpha=tau)

    def save(self, path=None):
        path = path or self.checkpoint_path
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state"])
        self.target_net.load_state_dict(data["target_state"])
        self.steps_done = data.get("steps_done", 0)
