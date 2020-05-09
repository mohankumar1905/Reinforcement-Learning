import gym
env = gym.make('CartPole-v0')
env.reset()
done = False
action_count = 1
while not done:
    action_count+=1
    observation, reward, done, _ = env.step(env.action_space.sample())
print(action_count)
