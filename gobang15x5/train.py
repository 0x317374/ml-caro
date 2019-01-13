import sys
import os

sys.path.append('..')
from board_game_base.coach import Coach
from gobang15x5.game import GobangGame as Game
from gobang15x5.nnet_base import NNetWrapper as nn
from board_game_base.utils import *

temp_folder = "../drive/temp/gobang15x5/"
checkpoint_file = "best.pth.tar"

args = dotdict({
  "num_iters": 1000,
  "num_eps": 100,
  "temp_threshold": 15,
  "update_threshold": 0.6,
  "maxlen_of_queue": 50000,
  "num_MCTS_sims": 25,
  "arena_compare": 40,
  "cpuct": 1,
  "checkpoint": temp_folder,
  "load_model": os.path.isfile("{}{}".format(temp_folder, checkpoint_file)),
  "load_folder_file": (temp_folder, checkpoint_file),
  "num_iters_for_train_examples_history": 20,
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
