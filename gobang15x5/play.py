import sys

sys.path.append('..')
from board_game_base.arena import Arena
from board_game_base.mcts import MCTS
from gobang15x5.game import GobangGame, display
from gobang15x5.players import *
from gobang15x5.nnet_base import NNetWrapper as NNet

import numpy as np
from board_game_base.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = GobangGame()

# all players
# rp = RandomPlayer(g).play
hp = HumanGobangPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('../drive/temp/', 'best.pth.tar')
args1 = dotdict({ 'num_MCTS_sims': 50, 'cpuct': 1.0 })
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.get_action_prob(x, temp = 0))

# n2 = NNet(g)
# n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
# args2 = dotdict({'num_MCTS_sims': 25, 'cpuct':1.0})
# mcts2 = MCTS(g, n2, args2)
# n2p = lambda x: np.argmax(mcts2.get_action_prob(x, temp=0))

# noinspection PyUnresolvedReferences
arena = Arena(n1p, hp, g, display = display)
print(arena.play_games(2, verbose = True))
