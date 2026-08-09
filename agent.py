"""
Agente DQN (Deep Q-Network) — a "IA" propriamente dita.

- Rede convolucional simples (estilo Nature DQN) recebe 4 frames
  empilhados em escala de cinza e decide: pular ou não pular.
- Replay buffer guarda experiências passadas pra treinar em lote.
- Epsilon-greedy: começa explorando bastante (ações aleatórias) e vai
  ficando cada vez mais "confiante" nas próprias decisões.
- Checkpoint salvo em disco -> se você fechar e abrir de novo, ela
  continua de onde parou (aprendizado incremental).
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
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

        self.head = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


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
                 lr=1e-4, gamma=0.99, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.checkpoint_path = checkpoint_path

        self.policy_net = QNetwork(frame_stack, n_actions).to(self.device)
        self.target_net = QNetwork(frame_stack, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 100_000

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            print(f"[agent] checkpoint carregado de {checkpoint_path} "
                  f"(steps_done={self.steps_done})")

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, explore=True):
        eps = self.epsilon() if explore else 0.0
        if explore and random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

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
            next_q = self.target_net(s2).max(dim=1)[0]
            target = r + (1 - done) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.steps_done += 1
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

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
