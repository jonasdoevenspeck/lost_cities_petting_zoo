#%%
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    #action = policy(observation, agent)
    #env.step(action)

#%%

