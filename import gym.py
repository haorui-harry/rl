import gym

# 创建环境
env = gym.make('CartPole-v1')

# 初始化环境
state = env.reset()

for episode in range(1000):
    state = env.reset()[0]  # 获取初始状态
    # env.render()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample() 
        next_state, reward, done, truncated, _ = env.step(action)
        
    
        state = next_state
        total_reward += reward
        




    # 打印训练信息
    print(f"Episode {episode + 1}/{1000}, Total Reward: {total_reward}")

env.close()