from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from const import Const
import os
import time
import pathlib
from policy_value_net_keras import PolicyValueNet
import pickle
import shutil


class TrainPipeline:
  def __init__(self, init_model = None):
    # params of the board and the game
    self.board_width = Const.board_width
    self.board_height = Const.board_height
    self.n_in_row = Const.n_in_row
    self.board = Board(width = self.board_width, height = self.board_height, n_in_row = self.n_in_row)
    self.game = Game(self.board)
    # training params
    self.learn_rate = 2e-3
    self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
    self.temp = 1.0  # the temperature param
    self.n_playout = 400  # num of simulations for each move
    self.c_puct = 5
    self.buffer_size = 10000
    self.batch_size = 512  # mini-batch size for training
    self.data_buffer = deque(maxlen = self.buffer_size)
    self.play_batch_size = 1
    self.epochs = 5  # num of train_steps for each update
    self.kl_targ = 0.02
    self.game_batch_num = 1000000
    self.best_win_ratio = 0.0
    # num of simulations used for the pure mcts, which is used as
    # the opponent to evaluate the trained policy
    self.pure_mcts_playout_num = 1000
    if init_model:
      # start training from an initial policy-value net
      self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file = init_model)
    else:
      # start training from a new policy-value net
      self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
    self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct = self.c_puct, n_playout = self.n_playout, is_selfplay = 1)
    self.batch = 0

  def get_equi_data(self, play_data):
    """
    augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
      for i in [1, 2, 3, 4]:
        # rotate counterclockwise
        equi_state = np.array([np.rot90(s, i) for s in state])
        equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
        extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        # flip horizontally
        equi_state = np.array([np.fliplr(s) for s in equi_state])
        equi_mcts_prob = np.fliplr(equi_mcts_prob)
        extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data

  def collect_selfplay_data(self, n_games = 1):
    """collect self-play data for training"""
    for i in range(n_games):
      winner, play_data = self.game.start_self_play(self.mcts_player, temp = self.temp, is_shown = 1)
      play_data = list(play_data)[:]
      self.episode_len = len(play_data)
      # augment the data
      play_data = self.get_equi_data(play_data)
      self.data_buffer.extend(play_data)

  def policy_update(self):
    """update the policy-value net"""
    mini_batch = random.sample(self.data_buffer, self.batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = self.policy_value_net.policy_value(state_batch)
    for i in range(self.epochs):
      loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
      new_probs, new_v = self.policy_value_net.policy_value(state_batch)
      kl = np.mean(np.sum(old_probs*(np.log(old_probs+1e-10)-np.log(new_probs+1e-10)), axis = 1))
      if kl>self.kl_targ*4:  # early stopping if D_KL diverges badly
        break
    # adaptively adjust the learning rate
    if kl>self.kl_targ*2 and self.lr_multiplier>0.1:
      self.lr_multiplier /= 1.5
    elif kl<self.kl_targ/2 and self.lr_multiplier<10:
      self.lr_multiplier *= 1.5
    explained_var_old = (1-np.var(np.array(winner_batch)-old_v.flatten())/np.var(np.array(winner_batch)))
    explained_var_new = (1-np.var(np.array(winner_batch)-new_v.flatten())/np.var(np.array(winner_batch)))
    print("- kl: {:.5f}, lr_multiplier: {:.3f}, loss: {}, entropy: {}, explained_var_old: {:.3f}, explained_var_new: {:.3f}".format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
    return loss, entropy

  def policy_evaluate(self, n_games = 10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct = self.c_puct, n_playout = self.n_playout)
    pure_mcts_player = MCTS_Pure(c_puct = 5, n_playout = self.pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    for i in range(n_games):
      winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player = i%2, is_shown = 0)
      win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1]+0.5*win_cnt[-1])/n_games
    print("- num_playouts: {}, win: {}, lose: {}, tie: {}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


def run(training_pipeline):
  """run the training pipeline"""
  try:
    train_start = time.time()
    for i in range(training_pipeline.batch, training_pipeline.game_batch_num):
      print("---\nBatch {}...".format(i+1))
      training_pipeline.collect_selfplay_data(training_pipeline.play_batch_size)
      print("- episode_len: {}".format(training_pipeline.episode_len))
      if len(training_pipeline.data_buffer)>training_pipeline.batch_size:
        loss, entropy = training_pipeline.policy_update()
        print("- loss: {}, entropy: {}".format(loss, entropy))
      # check the performance of the current model, and save the model params
      is_check_freq = (i+1)%Const.check_freq==0
      is_check_freq_best = (i+1)%Const.check_freq_best==0
      if is_check_freq:
        print("- Current self-play batch: {}".format(i+1))
        model_file = "./drive/models/{}_current_{}x{}_{}_{}.model".format(Const.train_core, training_pipeline.board_width, training_pipeline.board_height, training_pipeline.n_in_row, time.strftime("%Y-%m-%d_%H-%M"))
        training_pipeline.policy_value_net.save_model(model_file)
        shutil.copy2(model_file, "./drive/models/{}_current_{}x{}_{}.model".format(Const.train_core, Const.board_width, Const.board_height, Const.n_in_row))
        # save state
        save_state(training_pipeline)
      if is_check_freq_best:
        win_ratio = training_pipeline.policy_evaluate()
        if win_ratio>training_pipeline.best_win_ratio:
          print("- New best policy!!!!!!!!")
          training_pipeline.best_win_ratio = win_ratio
          # update the best_policy
          training_pipeline.policy_value_net.save_model("./drive/models/{}_best_{}x{}_{}_{}.model".format(Const.train_core, training_pipeline.board_width, training_pipeline.board_height, training_pipeline.n_in_row, time.strftime("%Y-%m-%d_%H-%M")))
          if training_pipeline.best_win_ratio==1.0 and training_pipeline.pure_mcts_playout_num<5000:
            training_pipeline.pure_mcts_playout_num += 1000
            training_pipeline.best_win_ratio = 0.0
      print("- done {} seconds!".format(time.time()-train_start))
      train_start = time.time()
      training_pipeline.batch = i
  except KeyboardInterrupt:
    print("---\nQuit....")
    time.sleep(2)


def save_state(training_pipeline):
  start_save = time.time()
  _policy = training_pipeline.mcts_player.mcts._policy
  policy_value_net = training_pipeline.policy_value_net
  training_pipeline.mcts_player.mcts._policy = None
  training_pipeline.policy_value_net = None
  pickle.dump(training_pipeline, open("./drive/others/{}_training_pipeline_{}x{}_{}.p".format(Const.train_core, Const.board_width, Const.board_height, Const.n_in_row), "wb"))
  training_pipeline.mcts_player.mcts._policy = _policy
  training_pipeline.policy_value_net = policy_value_net
  print("- save state {} seconds".format(time.time()-start_save))


def load_state():
  init_model = None
  model_name = "./drive/models/{}_current_{}x{}_{}.model".format(Const.train_core, Const.board_width, Const.board_height, Const.n_in_row)
  if os.path.isfile(model_name):
    init_model = model_name
  print("init_model:", init_model)
  state_file = "./drive/others/{}_training_pipeline_{}x{}_{}.p".format(Const.train_core, Const.board_width, Const.board_height, Const.n_in_row)
  if os.path.isfile(state_file):
    training_pipeline = pickle.load(open(state_file, "rb"))
    # training_pipeline.policy_value_net =
    if init_model:
      training_pipeline.policy_value_net = PolicyValueNet(training_pipeline.board_width, training_pipeline.board_height, model_file = init_model)
    else:
      training_pipeline.policy_value_net = PolicyValueNet(training_pipeline.board_width, training_pipeline.board_height)
    training_pipeline.mcts_player.mcts._policy = training_pipeline.policy_value_net.policy_value_fn
    return training_pipeline
  return TrainPipeline(init_model = init_model)


if __name__=="__main__":
  pathlib.Path('./drive/models').mkdir(parents = True, exist_ok = True)
  pathlib.Path('./drive/others').mkdir(parents = True, exist_ok = True)
  print(time.strftime("%Y-%m-%d %H:%M"))
  training_pipeline = load_state()
  run(training_pipeline)
