import numpy as np
from ems_environment import ComplexEMS as EMS_Env
from helper import ActionSpaceGenerator
from mcts_puct import uct_search, StateNode, MinMaxStats
from tqdm import tqdm
import time

class Trainer:
    """
    The Trainer class holds the basic logics for creating training samples,
    training the network in a reinforcement learning manner and evaluating the 
    current strength of the neural network.
    """
    def __init__(self, config, nets, env):
        self.config = config
        self.episodes = config.episodes
        self.training_epochs = config.training_epochs
        self.simulations = config.simulations
        self.environment_steps = config.environment_steps
        self.batch_size = config.batch_size
        self.nets = nets
        self.env = env
        self.action_size = config.action_size

    def calc_summed_value(self, values, visits):
        # needed to create the training data for the policy
        sum_val = 0
        for i, v in enumerate(visits):
            if v != 0:
                sum_val += values[i]
        return sum_val

    def calc_dirichlet(self, child_priors, dir_x=0.75, dir_alpha=0.3):
        # adds exploration noise to the policy
        priors = dir_x * child_priors + (1 - dir_x) * np.random.dirichlet([dir_alpha] * self.action_size)
        return priors

    def create_training_samples(self, env):
        states, next_states, rewards, actions, visits, values = [], [], [], [], [], []
        actions_taken = np.zeros(self.action_size)
        cum_r, good_action, bad_action = 0, 0, 0
        for ep in tqdm(range(self.episodes)):
            state = env.reset()
            done = False
            while not done:
                uct_node = uct_search(StateNode(state, env, self.nets, env.time, action_size=self.action_size),
                                      self.simulations,
                                      use_dirichlet=False,
                                      action_size=self.action_size)
                sum_visits = sum(uct_node.child_number_visits)
                print(uct_node.child_number_visits)
                action = np.argmax(uct_node.child_numer_visits)
                next_state, reward, done, info = env.step(action)
                root_value = self.calc_summed_value(uct_node.child_total_value,
                                                    uct_node.child_number_visits)/sum_visits
                states.append(state)
                next_states.append(next_state)
                rewards.append(reward)
                actions.append(self.to_one_hot(action))
                visits.append(uct_node.child_number_visits/sum_visits)
                values.append(root_value)
                state = next_state
                actions_taken[action] += 1
                cum_r += reward
                if reward == -0.1/5:
                    bad_action += 1
                else:
                    good_action += 1
        print(f"Actions: {actions_taken}\n"
              f"Policy:  {np.array(uct_node.child_number_visits/sum_visits).round(2)}\n"
              f"Root Value min: {np.round(min(values), 2)} max: {np.round(max(values), 2)}\n"
              f"Values: {np.array(uct_node.child_total_value/uct_node.child_number_visits).round(2)}\n"
              f"Reward/Episode: {np.round(cum_r/self.episodes, 2)}\n"
              f"Env Time: {env.time}\n"
              f"Bad Action %: {np.round(bad_action/(bad_action+good_action), 2)}")
        t_values = self.calc_t_values(rewards, values)
        # Minus 5 for correct bootstrap values!
        training_dict = {"states": states[:-5],
                         "next_states": next_states[:-5],
                         "rewards": rewards[:-5],
                         "actions": actions[:-5],
                         "visits": visits[:-5],
                         "values": t_values[:-5]}
        return training_dict

    def calc_t_values(self, rewards, root_values, td_steps=5, discount=0.99):
        # This function calculates the target q-values for a fixed step 
        # size into the future. It is not needed in the current build,
        # but necessary if we plan to make the current algorithm more like MuZero
        t_values = []
        for i, value in enumerate(root_values):
            bootstrap_index = i + td_steps
            if bootstrap_index < len(root_values):
                value = root_values[bootstrap_index] * discount ** td_steps
            else:
                value = 0
            for j, reward in enumerate(rewards[i:bootstrap_index]):
                value += reward * discount ** j
            t_values.append(value)
        return t_values

    def training_phase(self, batch, sample_weights):
        """
        This function uses past experiences and trains the network
        :param eval_train: The needed training samples to train the network
        :return: a training history object to see the losses
        """
        tack = time.time()
        print(f"Training network {self.training_epochs} times with {len(batch['env_states'])} samples. ")
        actions = np.array([self.config.asg.get_action(act)[0] for act in batch["actions"]])
        self.nets.fit([batch["env_states"], batch["det_states"], actions],
                                  [batch["env_next"], batch["rewards"]],
                                  epochs=self.training_epochs,
                                  batch_size=self.config.training_batch_size,
                                  sample_weight=sample_weights)
        # score, high_score = self.evaluate_current_iteration()
        self.nets.save_weights("weights.h5")
        tick = time.time()

        print(f"Training phase took {int(tick - tack)} seconds.")

    def to_one_hot(self, move):
        # utility for one-hot encoding. Currently not needed
        one_hot_vector = np.zeros(self.action_size)
        one_hot_vector[move] = 1
        return one_hot_vector

    def discount(self, window):
        # discount utility for MuZero like algorithm
        value = window[0]
        for i in range(1, 5):
            value += window[i] * 0.99 ** i
        return value

    def evaluate_current_iteration(self, eval_time=20000, env_steps=96, log_file="results_log.csv"):
        """

        :param high_score: a variable to determine if the weights of the current network get saved
        :param forecast: a list of forecast
        :param LOGFILE: flag, determines if an additional log_file is created
        :param PLOT: flag, determines if the results get plotted via matplotlib
        :return: the current highscore
        """
        test_env = self.env
        state = test_env.static_reset(eval_time)
        cum_r = 0
        actions_taken = np.zeros(self.action_size)
        min_max = MinMaxStats(init=True)
        for _ in tqdm(range(env_steps)):
            prior_state = state
            uct_node = uct_search(StateNode(state, test_env, self.nets, test_env.time, self.config, test_env.variables),
                                  self.simulations,
                                  self.config,
                                  use_dirichlet=False)

            action = np.argmax(uct_node.child_number_visits)

            state, reward, done, info = test_env.step(action, EVALUATION=True)
            if reward == -1:
                # only for debugging!
                print("Env state")
                print(prior_state)
                print("det state")
                print(uct_node.state.state_det)
                print("visits:")
                print(uct_node.child_number_visits)
                print("values:")
                print(uct_node.child_total_value)
                print("rewards:")
                print(uct_node.state.rewards)
                print("priors:")
                print(uct_node.state.child_priors)
            cum_r += reward
            actions_taken[action] += 1
            print(cum_r)
            # print(action)
            # print(f"Actions: {actions_taken}\n"
            #       f"Policy:  {uct_node.child_number_visits/sum(uct_node.child_number_visits)}\n"
            #       f"Values:  {uct_node.child_total_value/uct_node.child_number_visits}")
        if cum_r > min_max.high_score:
            print(f"\n\n--=== New highscore achieved: {cum_r}! ===--\n\n")
            # forecast_network.save_weights("./networks/best_forecast_weights.h5")
            # self.test_nets.save_weights("./checkpoints/evaluator_weights.h5")
#            self.nets = self.test_nets
            min_max.high_score = cum_r
            min_max.save()
        else:
            print(f"No new highscore, current performance: {cum_r}!")
        return cum_r, min_max.high_score

