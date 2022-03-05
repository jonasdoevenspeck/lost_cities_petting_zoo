class Expedition:

    def __init__(self):
        self.expeditions = {
            'green' : [],
            'blue' : [],
            'yellow' : [],
            'red' : [],
            'white' : []
        }

    def add_card(self,card):
        card_color = card['color']
        #card_val = card['val']
        #card = 
        self.expeditions[card_color].append(card)

    def can_build(self, cards):

        
        for card in cards:
            card_color = card['color']
            card_val = card['val']
            #check if color in hand is not yet started
            if len(self.expeditions[card_color]) == 0:
                return True

            #check if val in hand is boost and a boost is on top
            elif card_val == 'b':
                if self.expeditions[card_color][-1] == 'b':
                    return True

            #check if val in hand is larger than val of started exp
            elif card_val > self.expeditions[card_color][-1]:
                return True

            else:
                return False

    def get_possible_builds(self,cards):
        possible_builds = []
        for card in cards:
            card_color = card['color']
            card_val = card['val']
            if len(self.expeditions[card_color]) > 0:
                top_card_exp = self.expeditions[card_color][-1]
            #check if color in hand is not yet started
            if len(self.expeditions[card_color]) == 0:
                possible_builds.append(card)
                #return True

            #can always build on a boost
            elif top_card_exp['val'] == 'b':
                possible_builds.append(card)

            #check if val in hand is larger than val of started exp in case no boost on top
            elif isinstance(card_val, int) and (card_val > top_card_exp['val']):
                possible_builds.append(card)

        return possible_builds    


    def get_top_cards(self):
        top_cards = []

        for color in self.expeditions.keys():
            if len(self.expeditions[color]) > 0:
                top_card_exp = self.expeditions[color][-1]
                top_cards.append(top_card_exp)


        return top_cards  


    def get_total_score(self):
        start_bonus = 0
        total_score = 0
        for color in self.expeditions.keys():
            if len(self.expeditions[color])>0:
                col_score = 0
                multiplier = 1+self.expeditions[color].count('b')
                for card in self.expeditions[color]:
                    val = card['val']
                    if isinstance(val, int):
                        col_score += val
                col_score = (col_score-start_bonus)*multiplier
            else:
                col_score = 0
            total_score += col_score
        return total_score

class Hand:

    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        id = card['id']
        self.cards[:] = [d for d in self.cards if d.get('id') != id]



class Pile:

    def __init__(self):
        self.piles = {
            'green' : [],
            'blue' : [],
            'yellow' : [],
            'red' : [],
            'white' : []
        }

    def add_card(self,card):
        card_color = card['color']
        self.piles[card_color].append(card)

    def draw_card(self, card):
        id = card['id']
        color = card['color']
        self.piles[color][:] = [d for d in self.piles[color] if d.get('id') != id]
        return card

    def draw_card_by_color(self, color):
        assert len(self.piles[color])>0, f"No cards with color {color}"
        card = self.piles[color][-1]
        id = card['id']
        #color = card['color']
        self.piles[color][:] = [d for d in self.piles[color] if d.get('id') != id]
        return card

    def get_visible_cards(self):
        visible_cards = []
        for color in self.piles.keys():
            if len(self.piles[color])>0:
                visible_cards.append(self.piles[color][-1])

        return visible_cards


