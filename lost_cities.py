#%%
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

from gym import spaces
from classes import Hand, Expedition, Pile

#%% we need to make a custom environment


class CustomEnv(gym.Env):

    def __init__(self,):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        #self.action_space = spaces.Discrete(69)
        #self.observation_space = spaces.Discrete(960)


        self.action_space = spaces.Tuple((
                spaces.Discrete(2), # play or discard
                #spaces.Discrete(60), #60 possible cards to be played or discarded
                spaces.Discrete(8), #8 possible cards in hand to play or discard
                spaces.Discrete(2), #draw blind or from pile
                spaces.Discrete(5))) #draw yellow, blue, white, red or green

        # 0 means no card
        self.observation_space = spaces.Tuple((
            spaces.Box(low = 0, high = 60,shape=(1,5), dtype = np.int8), #cards ids on top of draw piles
            spaces.Box(low = 0, high = 60,shape=(1,5), dtype = np.int8), #cards ids on top of expeditions
            spaces.Box(low = 1, high = 60,shape=(1,8), dtype = np.int8), #cards ids in hand
        ))

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_player = 0
        total_cards = 60
        start_cards = 8

        self.colors = ['green','blue','yellow','red','white']
        cards = [2,3,4,5,6,7,8,9,10,'b','b','b']

        self.current_step = 0
        self.card_deck = []
        self.discard_deck = []
        self.expeditions = []
        id = 1
        for color in self.colors:
            for card in cards:
                self.card_deck.append({'id': id,'color': color, 'val': card})
                id +=1

        self.starting_deck = self.card_deck.copy()

        random.shuffle(self.card_deck)

        self.hands = [Hand(),Hand()]
        self.exps = [Expedition(),Expedition()]
        self.centre_pile = Pile()
        for player_idx in range(2):
            for card_idx in range(start_cards):
                self.hands[player_idx].add_card(self.card_deck.pop())
            #self.hands[self.current_player].add_card(self.card_deck.pop())

    def get_card_by_id(self, id):

        for card in self.starting_deck:
            if card['id'] == id:
                return card

    def get_id_by_card(self, card):

        for card in self.starting_deck:
            id = card['id']
            return id

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if len(self.card_deck) == 0:
            done = True
        else:
            done = False
        reward = self.exps[self.current_player].get_total_score()
        obs = self._next_observation()
        #print(obs)
        info = {}

        #switch players
        if self.current_player == 0:
            self.current_player = 1
        else:
            self.current_player = 0

        return obs, reward, done, info


    def _next_observation(self):

        visible_cards = self.centre_pile.get_visible_cards()
        visible_card_ids = []
        for card in visible_cards:
            visible_card_ids.append(card['id'])

        visible_cards_obs = np.zeros((1,5),dtype=np.int8)
        visible_cards_obs[0,0:len(visible_card_ids)] = np.asarray(visible_card_ids,dtype=np.int8)

        top_cards = self.exps[self.current_player].get_top_cards()
        top_cards_ids = []
        for card in top_cards:
            top_cards_ids.append(card['id'])

        top_cards_obs = np.zeros((1,5),dtype=np.int8)
        top_cards_obs[0,0:len(top_cards_ids)] = np.asarray(top_cards_ids,dtype=np.int8)

        hand_cards = self.hands[self.current_player].cards
        hand_cards_ids = []
        for card in hand_cards:
            hand_cards_ids.append(card['id'])
        hand_cards_obs = np.zeros((1,8),dtype=np.int8)
        hand_cards_obs[:] = np.asarray(hand_cards_ids,dtype=np.int8)

        obs = (visible_cards_obs,top_cards_obs,hand_cards_obs)
        return obs


    def _take_action(self,action):

        play_actions = ['build','discard']
        draw_actions = ['draw_blind','draw_discard']

    
        play_action_vec = action[0]
        card_id_played = action[1]
        draw_action_id = action[2]
        draw_pile_id = action[3]

        #determine here if actions are legal, if not, give negative reward
        #TODO: implement illegal moves, look up what to do in case
        #or provide action mask outside gym env

        print(action)

        possible_builds = self.exps[self.current_player].get_possible_builds(self.hands[self.current_player].cards)
        
        possible_build_ids = []
        for dic in possible_builds:
            possible_build_ids.append(dic['id'])

        hand_ids = []
        for dic in self.hands[self.current_player].cards:
            hand_ids.append(dic['id'])       

        if play_action_vec:
            play_action = 'build'
        else:
            play_action = 'discard'

        if play_action == 'build':
            print('build')
            #card_id_played = random.randint(0,len(possible_builds)-1)
            #played_card = self.get_card_by_id(card_id_played)
            played_card = self.hands[self.current_player].cards[card_id_played]
            self.exps[self.current_player].add_card(played_card) # add card to expedition
            self.hands[self.current_player].remove_card(played_card) # remove card from hand
        
        elif play_action == 'discard':
            print('discard')
            #card_id_played = random.randint(0,len(self.hands[self.current_player].cards)-1)
            #played_card = self.get_card_by_id(card_id_played)
            played_card = self.hands[self.current_player].cards[card_id_played]
            self.centre_pile.add_card(played_card) # add card to centre pile
            self.hands[self.current_player].remove_card(played_card) # remove card from hand


        # if no cards on discard deck, draw from blind deck
        visible_cards = self.centre_pile.get_visible_cards()
        
        if draw_action_id == 0:
            draw_action = 'draw_blind'
        elif draw_action_id == 1:
            draw_action = 'draw_discard'

        if draw_action == 'draw_blind':
            print('draw blind')
            new_card = self.card_deck.pop()
            self.hands[self.current_player].add_card(new_card)

        elif draw_action == 'draw_discard':
            print('draw discard')
            #card_id_draw = draw_card_id
            #new_card = self.get_card_by_id(draw_card_id)
            color = self.colors[draw_pile_id]
            #new_card = self.centre_pile.piles[color][-1]
            #need to replace card id by color pile here
            drawn_card = self.centre_pile.draw_card_by_color(color) # add card to centre pile
            self.hands[self.current_player].add_card(drawn_card) # remove card from hand 

    def check_if_action_valid(self,action):

        play_action_vec = action[0] # play or discard
        card_id_played = action[1] #60 possible cards to be played or discarded
        draw_action_id = action[2] #draw blind or from pile
        draw_pile_id = action[3] #draw yellow, blue, white, red or green

        if play_action_vec:
            play_action = 'build'
        else:
            play_action = 'discard'


        if draw_action_id == 0:
            draw_action = 'draw_blind'
        elif draw_action_id == 1:
            draw_action = 'draw_discard'

        
        #1st check can we play? discard always possible
        if play_action == 'discard':
            play_action_check = True
        elif play_action == 'build':
            possible_builds = self.exps[self.current_player].get_possible_builds(self.hands[self.current_player].cards)
            selected_card = self.hands[self.current_player].cards[card_id_played]
            if selected_card in possible_builds:
                play_action_check = True
            else:
                play_action_check = False


        #2: card id is card in hand so always possible
        # no, need to check if card id is in possible builds
        # but we do this above now
        
        #if selected_card in possible_builds:



        #3: draw blind or from pile: check is pile is not empty
        # this check is redundant since we also check for color next check 
        # however for now we keep it for informativeness
        if draw_action == 'draw_discard':
            if len(self.centre_pile.get_visible_cards())>0:
                draw_action_check1 = True
            else:
                draw_action_check1 = False           
        #4: is color available?

            color = self.colors[draw_pile_id]
            if len(self.centre_pile.piles[color])>0:
                draw_action_check2 = True
            else:
                draw_action_check2 = False           

            draw_action_check = draw_action_check1 and draw_action_check2

        elif draw_action == 'draw_blind':
                draw_action_check = True      

        

        if play_action_check and draw_action_check:
            return True
        else:
            return False



#%%

done = False
env = CustomEnv()
env.reset()
for step_nr in range(200):
    if done == False:
        print(f'step no. {step_nr}')
        action = env.action_space.sample()
        print('--------------')
        print('--------------')
        print(action)
        print('--------------')
        valid = env.check_if_action_valid(action)
        if valid:
            print('valid')
            obs, reward, done, info = env.step(action)
            print('reward',reward)
        else:
            print('invalid')


#%%


env = gym.make('LostCities-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# %%
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
# %%




#%%

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

#%%

n_actions = 420
# Get number of actions from gym action space
n_actions = env.action_space.n
policy_net = DQN(16, 60, n_actions).to(device)
target_net = DQN(16, 60, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

#%%

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

#%%

env = CustomEnv()


num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, reward, done, info = env.step(action)
    
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    #time.sleep(0.001)
    
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()