from otree.api import *




doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'end'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    anket_ = models.StringField(initial="99")
    finished = models.BooleanField(initial=0)


    
    

# PAGES
class Anket(Page):
    form_model = Player
    form_fields = ["anket_"]

class Finish(Page):
    @staticmethod
    def vars_for_template(player: Player):
        player.finished=1
    

page_sequence = [Anket, Finish]
