from board_game_base.coach import Coach
from tictoctoe.game import TicTacToeGame as Game
from tictoctoe.nnet_base import NNetWrapper as nn
from board_game_base.utils import *

args = dotdict({
  'num_iters': 1000,
  'num_eps': 100,
  'temp_threshold': 15,
  'update_threshold': 0.6,
  'maxlen_of_queue': 200000,
  'num_MCTS_sims': 25,
  'arena_compare': 40,
  'cpuct': 1,
  'checkpoint': '../drive/temp/',
  'load_model': False,
  'load_folder_file': ('../drive/temp/', 'best.pth.tar'),
  'num_iters_for_train_examples_history': 20,
})

if __name__=="__main__":
  g = Game()
  nnet = nn(g)
  if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
  c = Coach(g, nnet, args)
  if args.load_model:
    print("Load train_examples from file")
    c.load_train_examples()
  c.learn()
