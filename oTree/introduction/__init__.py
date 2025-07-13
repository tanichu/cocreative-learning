from otree.api import *



doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'introduction'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):        
    CrowdWorks_ID=models.StringField()
    sex=models.StringField(choices=["男性","女性","その他"])
    age=models.IntegerField(choices=[i for i in range(18,71)])
    is_accepted = models.BooleanField()
    #pplayer = BasePlayer.in_round(BasePlayer.round_number - 1)

    
    

# PAGES
class Introduction(Page):
    form_model = Player
    form_fields = ["is_accepted"]
    

class Disagree(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.is_accepted == 0

class Agree(Page):
    form_model = Player
    form_fields = ["CrowdWorks_ID","sex","age"]
    @staticmethod
    def is_displayed(player: Player):
        return player.is_accepted == 1

class Tutorial(Page):
    pass


class TCategorization(Page):
    pass
    

page_sequence = [Introduction, Disagree, Agree,Tutorial]
