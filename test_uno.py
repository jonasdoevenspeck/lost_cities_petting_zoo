#%%
from pettingzoo.classic.rlcard_envs import uno as uno
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

env = uno.env()
# %%
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    #action = policy(observation, agent)
    #env.step(action)
# %%

#%%
#env = gym.make('CartPole-v1')

model = PPO2(MlpPolicy, env, verbose=0)
# %%
