#%%
from gym import spaces
import numpy as np
import random
import functools
import  gym
from gym.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v5
import supersuit as ss

from classes import Hand, Expedition, Pile
#%%
printt = False

def env():
    env = raw_env()
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv, EzPickle):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {"name": "lost_cities",'is_parallelizable': True,'render.modes': ['human']}

    def __init__(self):
        EzPickle.__init__(self)

        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(2))))
        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = dict(zip(self.possible_agents,[spaces.Discrete(160)]*2))
        self.observation_spaces = dict(zip(self.possible_agents,[spaces.Box(low = 0, high = 60,shape=(1,18), dtype = np.int8)]*2))


        self.num_moves = 0


        self.current_player = 0
        self.total_cards = 60
        self.start_cards = 8

        self.colors = ['green','blue','yellow','red','white']
        self.card_vals = [2,3,4,5,6,7,8,9,10,'b','b','b']


        #self.observation = self._next_observation(0)
        

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
            print(f"Current state: Agent1: {self.hands[0].cards} , Agent2: {self.hands[1].cards}")
            print(f"exp 1 {self.exps[0].expeditions}")
            print(f"exp 2 {self.exps[1].expeditions}")
            #print()
        else:
            string = "Game over"
        #print(string)


    def observe(self, agent):
        #print(agent)

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

        # self.agents = self.possible_agents[:]
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: {} for agent in self.agents}



        #self.observations = {agent: self._next_observation(agent_idx) for agent_idx, agent in enumerate(self.agents)}
        #self.observations = self._next_observation()

        #self._agent_selector = agent_selector(self.agents)
        #self.agent_selection = self._agent_selector.next()
        self.current_step = 0
        self.card_deck = []
        self.discard_deck = []
        #self.expeditions = []
        id = 1
        for color in self.colors:
            for card in self.card_vals:
                self.card_deck.append({'id': id,'color': color, 'val': card})
                id +=1

        self.starting_deck = self.card_deck.copy()

        random.shuffle(self.card_deck)

        self.hands = [Hand(),Hand()]
        self.exps = [Expedition(),Expedition()]
        self.centre_pile = Pile()
        for player_idx in range(2):
            for card_idx in range(self.start_cards):
                self.hands[player_idx].add_card(self.card_deck.pop())


        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.has_reset = True
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

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

        #5: did we not just the discard the card?

            draw_action_check = draw_action_check1 and draw_action_check2

            if play_action == 'discard' and draw_action == 'draw_discard':
                selected_card = self.hands[self.current_player].cards[card_id_played]
                discard_color = selected_card['color']
                draw_color = self.colors[draw_pile_id]
                if draw_color != discard_color:
                    draw_action_check3 = True
                else:
                    draw_action_check3 = True


                draw_action_check = draw_action_check1 and draw_action_check2 and draw_action_check3

        elif draw_action == 'draw_blind':
                draw_action_check = True      

        

        if play_action_check and draw_action_check:
            return True
        else:
            return False


    def _take_action(self,action):
        #print(action)
        play_actions = ['build','discard']
        draw_actions = ['draw_blind','draw_discard']

    
        play_action = self.action_vec_dict[action]['play_action']
        card_id_played = self.action_vec_dict[action]['card_id_played']
        draw_action = self.action_vec_dict[action]['draw_action']
        draw_pile_id = self.action_vec_dict[action]['draw_pile_id']

        #determine here if actions are legal, if not, give negative reward
        #TODO: implement illegal moves, look up what to do in case
        #or provide action mask outside gym env

        #print(action)

        possible_builds = self.exps[self.current_player].get_possible_builds(self.hands[self.current_player].cards)
        
        possible_build_ids = []
        for dic in possible_builds:
            possible_build_ids.append(dic['id'])

        hand_ids = []
        for dic in self.hands[self.current_player].cards:
            hand_ids.append(dic['id'])       


        if play_action == 'build':
            #print('building')
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
            #print('length',len(self.card_deck))
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


    def step(self, action):
        # Execute one time step within the environment
        if printt:
            print('step done 1', self.dones)
            print('action', action)
            print('agent',self.agent_selection)
            print('nr cards', len(self.card_deck))
        # if len(self.card_deck)<1 or self.dones[self.agent_selection]:
        #     self.dones['player_0'] = True
        #     self.dones['player_1'] = True

        #     #action = None
        #     return self._was_done_step(action)

        if self.dones[self.agent_selection]:
        #    print('step done 2', self.dones[self.agent_selection])
        #    self.card_out = True
            action = None
            return self._was_done_step(action)

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        valid = self.check_if_action_valid(action)
        if len(self.card_deck)<1:
            self.done = True
        if valid:
            #print('a')
            self._take_action(action)
        else:
            #if printt:
            print('invalid')
            next_reward = -1

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            if printt:
                print('is last')
            # rewards for all agents are placed in the .rewards dictionary
            if valid:
                self.rewards[self.agents[0]] = self.exps[0].get_total_score()
                self.rewards[self.agents[1]] = self.exps[1].get_total_score()
            else:
                self.rewards[self.agents[0]] = -10
                self.rewards[self.agents[1]] = -10
            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            
            # observe the current state
            #for i_idx, i in enumerate(self.agents):
            #    self.observations[i] = self._next_observation(i_idx)
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            #self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._next_observation(i)
            # no rewards are allocated until both players give an action
            self._clear_rewards()
        self.dones = {agent: len(self.card_deck)<2 for agent in self.agents}
        if self._agent_selector.is_last():
            self.dones = dict(zip(self.agents, [self.done for _ in self.agents]))

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

#%%

from pettingzoo.utils.conversions import aec_to_parallel

env = env()
#env = wrappers.flatten_observation(env)
penv = aec_to_parallel(env)
penv = ss.pettingzoo_env_to_vec_env_v1(penv)
#%%
penv = ss.concat_vec_envs_v1(penv, 1, num_cpus=1, base_class='stable_baselines3')


#%%
model = PPO(MlpPolicy, penv, verbose=1,n_steps=200,batch_size=100,n_epochs=10)

#%%
model.learn(total_timesteps=1000000)
model.save('policy_reward_min_10_steps')



#%%

play_actions = ['build','discard']
card_ids_played = list(range(8))
draw_actions = ['draw_blind','draw_discard']
draw_pile_ids = list(range(5))
action_id = 0
action_vec_dict = {}

for play_action in play_actions:
    for card_id_played in card_ids_played:
        for draw_action in draw_actions:
            for draw_pile_id in draw_pile_ids:
                action_vec_dict[action_id] = {'play_action':play_action,'card_id_played':card_id_played,'draw_action': draw_action, 'draw_pile_id':draw_pile_id}
                action_id += 1

model = PPO.load('policy_reward_min_10_steps')

env.reset()
steps = 0
for agent in env.agent_iter():
    steps +=1
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    play_action = action_vec_dict[act]['play_action']
    card_id_played = action_vec_dict[act]['card_id_played']
    draw_action = action_vec_dict[act]['draw_action']
    draw_pile_id = action_vec_dict[act]['draw_pile_id']
    print('action id', act)
    print('play action', play_action)
    print('card_id_played', card_id_played)
    print('draw_action', draw_action)
    print('draw_pile_id', draw_pile_id)
    print('------------------------------------')
    print('------------------------------------')
    env.step(act)
    env.render()
    if steps > 100:
        break

#always 




# %%
