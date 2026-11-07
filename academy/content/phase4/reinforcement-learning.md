---
{
  "id": "p4-reinforcement-learning",
  "title": "Reinforcement Learning",
  "phase": 4,
  "difficulty": "advanced",
  "estimated_hours": 16,
  "objectives": [
    "Understand the RL framework: agent, environment, state, action, reward",
    "Implement Q-learning and policy gradient methods",
    "Apply RL to control and sequential decision problems",
    "Recognize when RL is appropriate vs supervised learning"
  ],
  "concepts": [
    "MDP formulation: states, actions, transitions, rewards",
    "Discount factor gamma and return",
    "Value functions: V(s), Q(s,a)",
    "Q-learning (off-policy TD)",
    "SARSA (on-policy TD)",
    "Policy gradient: REINFORCE",
    "Actor-critic methods",
    "Exploration vs exploitation: epsilon-greedy, UCB"
  ],
  "formulas": [
    {"name": "Return", "latex": "G_t = \\sum_{k=0}^{\\infty} \\gamma^k r_{t+k+1}", "note": "Discounted cumulative reward."},
    {"name": "Q-learning update", "latex": "Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]", "note": "Temporal difference learning."},
    {"name": "Bellman equation", "latex": "V(s) = \\max_a \\sum_{s',r} P(s',r|s,a)[r + \\gamma V(s')]", "note": "Optimal value function."},
    {"name": "Policy gradient", "latex": "\\nabla J = \\mathbb{E}[\\nabla \\log \\pi(a|s) \\cdot G_t]", "note": "REINFORCE estimator."}
  ],
  "references": [
    "Sutton & Barto 'Reinforcement Learning: An Introduction' (free)",
    "Goodfellow et al. ch. 13",
    "OpenAI Gymnasium documentation",
    "Stable Baselines3 documentation"
  ]
}
---

## Penjelasan

Reinforcement Learning (RL) adalah paradigma di mana agen belajar kebijakan (policy) melalui interaksi dengan lingkungan — bukan dari data berlabel, melainkan dari *reward* yang diterima setelah mengambil tindakan. RL cocok untuk sequential decision problems: game playing, robotics, resource allocation.

**Framework.** Agent mengamati *state* s_t, memilih *action* a_t, menerima *reward* r_t dan state baru s_{t+1}. Tujuan: maksimalkan expected cumulative discounted reward G_t = Σ γ^k r_{t+k+1}. γ (gamma) ∈ [0,1) mengontrol seberapa jauh reward masa depan dihitung.

**Value functions.** V(s) = expected return dari state s mengikuti policy π. Q(s,a) = expected return dari state s mengambil action a. Bellman equation mendefinisikan hubungan rekursif antara value dan reward — fondasi semua RL algorithms.

**Q-learning.** Off-policy temporal difference: update Q(s,a) menuju r + γ max_a' Q(s',a'). Belajar *nilai* action terbaik tanpa mengikuti policy saat ini (exploration vs exploitation). Converge ke optimal Q* jika semua state-action pairs dikunjungi cukup.

**SARSA.** On-policy: update menuju r + γ Q(s',a') di mana a' adalah action yang *benar-benar* diambil policy — lebih konservatif, belajar nilai policy yang sedang dijalankan.

**Policy gradient.** Alih-alih belajar value, langsung optimasi policy parameters: ∇J = E[∇log π(a|s) · G_t]. REINFORCE adalah contoh paling sederhana; high variance tapi tidak terbatas pada discrete actions.

**Actor-critic.** Kombinasi: actor (policy) + critic (value function). Critic mengurangi variance policy gradient dengan baseline (advantage = G - V(s)).

## Konsep Penting

1. **Exploration vs exploitation** — epsilon-greedy: dengan prob ε pilih random action (explore), else greedy (exploit). UCB: pilih action dengan bonus uncertainty.
2. **Reward shaping** — reward design sangat memengaruhi learning; sparse rewards sulit; dense rewards bisa dimanipulasi.
3. **Credit assignment** — reward datang terlambat; agent harus assign ke action mana yang menyebabkan reward.
4. **RL vs supervised** — supervised belajar dari (input, output) pairs; RL belajar dari (state, action, reward) tuples — tidak ada "correct action", hanya reward signal.
5. **Sample inefficiency** — RL butuh jutaan interaksi; simulasi/digital twin membantu.

## Formula

- Return: G_t = Σ γ^k r_{t+k+1}.
- Q-learning: Q += α[r + γ max Q(s',a') - Q].
- Bellman optimality: V*(s) = max_a Σ P[r+γV*(s')].
- Policy gradient: ∇J = E[∇log π(a|s) · G_t].

## Contoh Kasus

Q-learning untuk grid world 4x4: states = posisi agent, actions = {up,down,left,right}, reward = +1 di goal, -1 di trap, -0.01 per step. Agent belajar policy optimal setelah ~1000 episodes. Atau CartPole: state = [pos, vel, angle, ang_vel], actions = {left, right}, reward = +1 per step survive.

## Implementasi

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, rng=None):
        rng = rng or np.random.default_rng()
        if rng.random() < self.epsilon:
            return rng.integers(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

# simple grid world
def grid_world_episode(agent, env, rng):
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.select_action(state, rng)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            return total_reward

# policy gradient (REINFORCE)
def reinforce_episode(env, policy_fn, optimizer, gamma=0.99):
    states, actions, rewards = [], [], []
    state = env.reset()
    while True:
        action, log_prob = policy_fn(state)
        next_state, reward, done = env.step(action)
        states.append(state); actions.append(action); rewards.append(reward)
        state = next_state
        if done: break
    # compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # policy gradient update would go here
    return sum(rewards)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "In RL, the agent learns through:",
    "options": [
      "Trial and error, receiving rewards for actions",
      "Labeled training data",
      "Supervised examples",
      "Unsupervised clustering"
    ],
    "answer": 0,
    "explanation": "RL = learning from reward signals through environment interaction.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "concept",
    "question": "The discount factor gamma controls:",
    "options": [
      "How much future rewards are valued vs immediate rewards",
      "Learning rate",
      "Exploration rate",
      "State space size"
    ],
    "answer": 0,
    "explanation": "gamma close to 1 = far-sighted; close to 0 = myopic.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Q-learning is 'off-policy' because:",
    "options": [
      "It learns optimal Q-values regardless of the policy being followed",
      "It doesn't use a policy",
      "It only works offline",
      "It uses no rewards"
    ],
    "answer": 0,
    "explanation": "Q-learning targets max_a Q(s',a') — learns about optimal policy while exploring.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma)` returning updated Q-table (modify in place or return new).",
    "starter_code": "import numpy as np\n\ndef q_learning_update(Q, state, action, reward, next_state, done, alpha=0.1, gamma=0.99):\n    Q = Q.copy()\n    if done:\n        target = reward\n    else:\n        target = reward + gamma * np.max(Q[next_state])\n    pass\n    return Q\n",
    "function": "q_learning_update",
    "tests": [
      {"call": "q_learning_update(np.zeros((2,2)),0,0,1.0,1,False,0.5,0.9)[0,0]", "expected": 0.5, "tolerance": 0.01}
    ],
    "hints": ["Q[state,action] += alpha * (target - Q[state,action])"],
    "explanation": "Temporal difference update.",
    "points": 2
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "Compare RL vs supervised learning for training a self-driving car: what data does each need, what are the failure modes, and when would you choose RL over imitation learning?",
    "rubric": [
      "RL needs reward function + simulator/real environment; supervised needs expert demonstrations",
          "RL can discover novel strategies but is sample-inefficient and unsafe to explore",
      "Supervised safer but limited to expert behavior",
      "RL for optimization beyond human performance; imitation for safe baseline"
        ],
    "points": 2
  }
]
```

## Referensi

- Sutton & Barto, 'Reinforcement Learning: An Introduction' (free online)
- Goodfellow et al. ch. 13
- OpenAI Gymnasium documentation
- Stable Baselines3 (SB3) documentation
- 'Deep Reinforcement Learning' by Arulkumaran et al.
