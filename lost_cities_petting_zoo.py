#%%
from gym import spaces
import numpy as np
import random
import functools
import  gym

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v5
import supersuit as ss

from classes import Hand, Expedition, Pile
#%%
def env():
    env = raw_env()
    env = wrappers.OrderEnforcingWrapper(env)
    return env


NUM_ITERS = 100

class raw_env(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {"name": "lost_cities"}

    def __init__(self):

        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        self.action_spaces = {agent: spaces.MultiDiscrete([2,8,2,5])   for agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Box(low = 0, high = 60,shape=(1,18), dtype = np.int8) for agent in self.possible_agents}
        self.reset()
        self.observation = self._next_observation(0)
        

    def get_card_by_id(self, id):
        for card in self.starting_deck:
            if card['id'] == id:
                return card

    def get_id_by_card(self, card):
        for card in self.starting_deck:
            id = card['id']
            return id


    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]


    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if len(self.agents) == 2:
            string = ("Current state: Agent1: {} , Agent2: {}".format(1,2))
        else:
            string = "Game over"
        print(string)


    def observe(self, agent):
        print(agent)

        #action_mask = np.zeros(9, 'int8')
        #for i in legal_moves:
        #    action_mask[i] = 1

        if agent == 'player_0':
            player = 0
        elif agent == 'player_1':
            player = 1

        return self._next_observation(player)


    def close(self):
        pass


    def _next_observation(self,current_player):

        self.current_player = current_player
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

        obs = np.concatenate([visible_cards_obs,top_cards_obs,hand_cards_obs],axis=1)
        return obs

    def reset(self):

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.num_moves = 0


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

        self.observations = {agent: self._next_observation(agent_idx) for agent_idx, agent in enumerate(self.agents)}
        #self.observations = self._next_observation()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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


    def step(self, action):
        # Execute one time step within the environment

        if self.dones[self.agent_selection]:
            return self._was_done_step(action)\

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        valid = self.check_if_action_valid(action)
        
        if valid:
            #print('a')
            self._take_action(action)
        else:
            next_reward = -1

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            if valid:
                self.rewards[self.agents[0]] = self.exps[0].get_total_score()
                self.rewards[self.agents[1]] = self.exps[1].get_total_score()
            else:
                self.rewards[self.agents[0]] = -1
                self.rewards[self.agents[1]] = -1
            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: len(self.card_deck)<1 for agent in self.agents}

            # observe the current state
            for i_idx, i in enumerate(self.agents):
                self.observations[i] = self._next_observation(i_idx)
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            #self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._next_observation(i)
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

            
        
#%%


#%%

env = env()



#%%
from pettingzoo.test import api_test
api_test(env, num_cycles=1000, verbose_progress=False)
#%%

env = ss.pettingzoo_env_to_vec_env_v1(env)

#%%
model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
model.learn(total_timesteps=20000)
model.save('policy')



#%%
model = PPO.load('policy')

env.reset()
for agent in env.agent_iter():
   obs, reward, done, info = env.last()
   act = model.predict(obs, deterministic=True)[0] if not done else None
   env.step(act)
   env.render()




