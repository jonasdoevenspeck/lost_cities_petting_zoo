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


        # self.action_space = spaces.Tuple((
        #         spaces.Discrete(2), # play or discard
        #         #spaces.Discrete(60), #60 possible cards to be played or discarded
        #         spaces.Discrete(8), #8 possible cards in hand to play or discard
        #         spaces.Discrete(2), #draw blind or from pile
        #         spaces.Discrete(5))) #draw yellow, blue, white, red or green

        self.action_space = spaces.Discrete(160)


        # 0 means no card
        self.observation_space = spaces.Tuple((
            spaces.Box(low = 0, high = 60,shape=(1,5), dtype = np.int8), #cards ids on top of draw piles
            spaces.Box(low = 0, high = 60,shape=(1,5), dtype = np.int8), #cards ids on top of expeditions
            spaces.Box(low = 1, high = 60,shape=(1,8), dtype = np.int8), #cards ids in hand
        ))

        play_actions = ['build','discard']
        card_ids_played = list(range(8))
        draw_actions = ['draw_blind','draw_discard']
        draw_pile_ids = list(range(5))
        self.done = False
        action_id = 0
        self.action_vec_dict = {}
        self.reset()
        for play_action in play_actions:
            for card_id_played in card_ids_played:
                for draw_action in draw_actions:
                    for draw_pile_id in draw_pile_ids:
                        self.action_vec_dict[action_id] = {'play_action':play_action,'card_id_played':card_id_played,'draw_action': draw_action, 'draw_pile_id':draw_pile_id}
                        action_id += 1



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
        valid = self.check_if_action_valid(action)
        if valid:
            self._take_action(action)
            reward = 0
        else:
            reward = -5

        self.current_step += 1
        if len(self.card_deck) == 0:
            done = True
            reward = self.exps[self.current_player].get_total_score()
        else:
            done = False
        
        obs = self._next_observation()
        #print(obs)
        info = {}

        #switch players
        if 0:
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

        obs = torch.tensor(np.concatenate((visible_cards_obs,top_cards_obs,hand_cards_obs),axis=1))[0].to('cuda').float().unsqueeze(0)
        return obs


    def _take_action(self,action):

        play_actions = ['build','discard']
        draw_actions = ['draw_blind','draw_discard']



        play_action = self.action_vec_dict[action]['play_action']
        card_id_played = self.action_vec_dict[action]['card_id_played']
        draw_action = self.action_vec_dict[action]['draw_action']
        draw_pile_id = self.action_vec_dict[action]['draw_pile_id']

        #print('player',self.current_player)
        #print(play_action, card_id_played)
        #print(draw_action, draw_pile_id)

        #determine here if actions are legal, if not, give negative reward
        #TODO: implement illegal moves, look up what to do in case
        #or provide action mask outside gym env

        #(action)

        possible_builds = self.exps[self.current_player].get_possible_builds(self.hands[self.current_player].cards)
        
        possible_build_ids = []
        for dic in possible_builds:
            possible_build_ids.append(dic['id'])

        hand_ids = []
        for dic in self.hands[self.current_player].cards:
            hand_ids.append(dic['id'])       


        if play_action == 'build':
            #print('build')
            #card_id_played = random.randint(0,len(possible_builds)-1)
            #played_card = self.get_card_by_id(card_id_played)
            played_card = self.hands[self.current_player].cards[card_id_played]
            self.exps[self.current_player].add_card(played_card) # add card to expedition
            self.hands[self.current_player].remove_card(played_card) # remove card from hand
        
        elif play_action == 'discard':
            #print('discard')
            #card_id_played = random.randint(0,len(self.hands[self.current_player].cards)-1)
            #played_card = self.get_card_by_id(card_id_played)
            played_card = self.hands[self.current_player].cards[card_id_played]
            self.centre_pile.add_card(played_card) # add card to centre pile
            self.hands[self.current_player].remove_card(played_card) # remove card from hand


        # if no cards on discard deck, draw from blind deck
        visible_cards = self.centre_pile.get_visible_cards()

        if draw_action == 'draw_blind':
            #print('draw blind')
            new_card = self.card_deck.pop()
            self.hands[self.current_player].add_card(new_card)

        elif draw_action == 'draw_discard':
            #print('draw discard')
            #card_id_draw = draw_card_id
            #new_card = self.get_card_by_id(draw_card_id)
            color = self.colors[draw_pile_id]
            #new_card = self.centre_pile.piles[color][-1]
            #need to replace card id by color pile here
            drawn_card = self.centre_pile.draw_card_by_color(color) # add card to centre pile
            self.hands[self.current_player].add_card(drawn_card) # remove card from hand 

    def check_if_action_valid(self,action):

        play_action = self.action_vec_dict[action]['play_action']
        card_id_played = self.action_vec_dict[action]['card_id_played']
        draw_action = self.action_vec_dict[action]['draw_action']
        draw_pile_id = self.action_vec_dict[action]['draw_pile_id']

        
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

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(18, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 160)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.softmax(self.fc3(x))

        return x



#inp = torch.tensor(obs[0])
#%%

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

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

#%%
n_actions = 160
device = 'cuda'
policy_net = DQN().to(device)
target_net = DQN().to(device)

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

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
            #print(state)
            policy_net.eval()

            action_out = policy_net(state)
            #print(action_out)
            action = action_out.max(1)[1].view(1, 1) 
            #print(action)
            policy_net.train()

            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


#%%

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


#%%
env = CustomEnv()
episode_durations = []
num_episodes = 500
for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and state
    env.reset()
    #last_obs = env._next_observation()
    #current_obs = env._next_observation()
    next_state = env._next_observation()
    for t in count():
        # Select and perform an action
        action = select_action(next_state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        
        last_state = next_state
        #print('last state', last_state)
        #current_obs = env._next_observation()
        next_state = env._next_observation()
        #print('next state', next_state)
        # Store the transition in memory
        memory.push(last_state, action, next_state, reward)

        # Move to the next state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            #print('done')
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# %%
target_net.eval()
done = False
#env = CustomEnv()
env.reset()
for step_nr in range(10):
    if done == False:
        print(f'step no. {step_nr}')
        obs = env._next_observation()
        action = target_net(obs).max(1)[1].view(1, 1).item()
        print('--------------')
        print('--------------')
        print(action)

        play_action = env.action_vec_dict[action]['play_action']
        card_id_played = env.action_vec_dict[action]['card_id_played']
        draw_action = env.action_vec_dict[action]['draw_action']
        draw_pile_id = env.action_vec_dict[action]['draw_pile_id']

        #print('player',self.current_player)
        print(play_action, card_id_played)
        print(draw_action, draw_pile_id)

        print('--------------')


        valid = env.check_if_action_valid(action)
        if valid:
            print('valid')
            obs, reward, done, info = env.step(action)
            print('reward',reward)
        else:
            print('invalid')

# %%
done = False
env = CustomEnv()
env.reset()
for step_nr in range(10):
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

# %%\
from typing import Optional
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce

class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
