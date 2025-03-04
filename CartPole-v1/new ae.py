import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定义深度Q网络（DQN）
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 两层全连接网络
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)

# 超参数设置
ENV_NAME = "CartPole-v1"
GAMMA = 0.99  # 折扣因子
LR = 1e-3  # 学习率
BATCH_SIZE = 64  # 批处理大小
REPLAY_BUFFER_SIZE = 10000  # 经验回放的容量
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率
EPSILON_START = 1.0  # epsilon 贪婪策略的起始值
EPSILON_END = 0.01  # epsilon 贪婪策略的结束值
EPSILON_DECAY = 500  # epsilon 衰减的步数

# 初始化环境、网络和优化器
# env = gym.make(ENV_NAME,render_mode="human")
env = gym.make(ENV_NAME)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# 经验回放缓冲区
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# epsilon 贪婪策略
epsilon = EPSILON_START

def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return env.action_space.sample()  # 随机选择动作
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = policy_net(state_tensor)
            return torch.argmax(q_values).item()

def calculate_custom_reward(state, done):
    """
    设计一个新的奖励机制，根据状态来计算奖励。
    - 角度小（越接近零），奖励越高。
    - 位置越接近零，奖励越高。
    - 如果杆子倾斜过大，给予较大的惩罚。
    """
    angle = state[2]  # 获取杆子的角度
    position = state[0]  # 获取小车的水平位置

    # 角度奖励：越接近零，奖励越大
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.15)

    # 位置奖励：越接近零，奖励越大
    position_reward = max(0.0, 1.0 - abs(position) / 2.4)

    # 如果杆子倾斜过大，则给较大的负奖励
    if abs(angle) > 0.2:
        return -1.0  # 杆子倾斜过大，回合结束

    # 如果杆子倾斜的角度较小，给予适度的奖励
    return angle_reward + position_reward

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    states = torch.stack(batch[0]).to(device)
    actions = torch.tensor(batch[1]).to(device)
    rewards = torch.tensor([calculate_custom_reward(state, done) for state, done in zip(batch[0], batch[4])]).to(device)
    next_states = torch.stack(batch[3]).to(device)
    dones = torch.tensor(batch[4]).to(device)

    # 计算当前状态的Q值
    state_action_values = policy_net(states).gather(1, actions.unsqueeze(1))

    # 计算下一个状态的最大Q值（使用目标网络）
    next_state_values = target_net(next_states).max(1)[0].detach()
    
    # 计算期望的Q值
    expected_state_action_values = rewards + (GAMMA * next_state_values * (1 - dones.to(torch.float32)))

    # 计算损失
    loss = nn.MSELoss()(state_action_values.squeeze(1), expected_state_action_values)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训练循环
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()[0]  # 获取初始状态
    # env.render()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        
        # 存储经验到回放缓冲区
        replay_buffer.push((torch.tensor(state, dtype=torch.float32), action, reward,
                            torch.tensor(next_state, dtype=torch.float32), done))

        # 更新状态
        state = next_state
        total_reward += reward
        
        # 优化模型
        optimize_model()

    # 逐步减少 epsilon
    epsilon = max(EPSILON_END, epsilon * (1 - 1 / EPSILON_DECAY))

    # 每隔一定步数更新目标网络
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 打印训练信息
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()
