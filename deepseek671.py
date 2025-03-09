import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_every = 100
        self.steps = 0
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

# 自定义奖励函数
def calculate_reward(next_state, done, terminated):
    if done:
        if terminated:  # 杆子倒下或小车出界
            return -10.0
        else:           # 正常完成500步
            return 10.0
    else:
        # 状态参数分解
        pos, vel, angle, ang_vel = next_state
        # 角度奖励（绝对值越小奖励越高）
        angle_ratio = abs(angle) / 0.2095  # 0.2095 radians ≈ 12 degrees
        angle_reward = 1.0 - angle_ratio**2
        # 位置奖励（绝对值越小奖励越高）
        pos_ratio = abs(pos) / 2.4
        pos_reward = 1.0 - pos_ratio**2
        # 总奖励组合
        return angle_reward * 0.7 + pos_reward * 0.3

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 500
max_steps = 5000
scores = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 计算自定义奖励
        custom_reward = calculate_reward(next_state, done, terminated)
        agent.save_experience(state, action, custom_reward, next_state, done)
        agent.learn()
        
        state = next_state
        total_reward += 1  # 原始环境奖励（每步+1）
        
        if done:
            break
    
    scores.append(total_reward)
    avg_score = np.mean(scores[-100:])
    print(f"Episode: {episode+1}, Score: {total_reward}, Avg Score: {avg_score:.1f}, Epsilon: {agent.epsilon:.2f}")
    
    if avg_score >= 495:
        print(f"Solved in {episode+1} episodes!")
        break

# 测试训练好的智能体
state, _ = env.reset()
total_reward = 0
while True:
    action = agent.act(state)
    next_state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += 1
    state = next_state
    if done:
        break
print(f"Test Score: {total_reward}")
env.close()