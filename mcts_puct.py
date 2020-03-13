import collections
import math
import numpy as np
from helper import MinMaxStats


class UCTNode:
    def __init__(self, state, move, config, parent=None):
        self.config = config
        self.action_size = config.action_size
        self.dummy = False
        self.state = state
        self.move = move
        self.is_expanded = False
        self.discount = 0.99
        self.parent = parent
        self.children = {i: DummyNode() for i in range(self.action_size)}
        self.child_total_value = np.zeros([self.action_size], dtype=np.float32)  # self.state.rewards
        self.child_number_visits = np.zeros([self.action_size], dtype=np.float32) # np.zeros([self.action_size], dtype=np.float32)
        self.q_scale = 1
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    def best_child(self, min_max):
        ucb_scores = {action: self.ucb_score(child, action, min_max) for action, child in self.children.items()}
        return max(ucb_scores, key=ucb_scores.get)

    def select_leaf(self, min_max):
        current = self
        while current.is_expanded:
            best_move = current.best_child(min_max)
            current = current.maybe_add_child(best_move)
        return current

    def expand(self):
        self.is_expanded = True

    def maybe_add_child(self, move):
        if self.children[move].dummy:
            self.children[move] = UCTNode(
                self.state.transition(move), move, self.config, parent=self)
        return self.children[move]

    def backup(self, value_estimate, min_max):
        current = self
        while current.parent is not None:  # and not current.state.terminal:
            current.number_visits += 1
            current.total_value += current.state.reward  # value_estimate
            # value_estimate = current.state.reward + value_estimate * self.discount
            min_max = min_max.update(current.child_total_value)
            current = current.parent
        return min_max

    def ucb_score(self, child, action, min_max):
        pb_c = math.log((self.number_visits + self.pb_c_base + 1) /
                        self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(self.number_visits) / (child.number_visits + 1)

        prior_score = pb_c * self.state.child_priors[action]
        value_score = min_max.normalize(child.total_value) * self.q_scale
        # value_score = child.total_value * self.q_scale
        return prior_score + value_score


def uct_search(state, simulations, config, use_dirichlet=False):
    root = UCTNode(state, move=0, config=config, parent=DummyNode())
    min_max = MinMaxStats()
    for _ in range(simulations):
        leaf = root.select_leaf(min_max)
        # child_priors, use_dirichlet = add_exploration_noise(leaf.state.child_priors, use_dirichlet)
        leaf.expand()
        min_max = leaf.backup(leaf.state.value_estimate, min_max)
    return root


def add_exploration_noise(child_priors, use_dirichlet, dir_x=0.75, dir_alpha=0.03):
    if use_dirichlet:
        priors = dir_x * child_priors + (1 - dir_x) * np.random.dirichlet([dir_alpha] * len(child_priors))
        return priors, False
    else:
        return child_priors, False


class StateNode:
    def __init__(self, state_env, env, r_net, time, config, variables, reward=0, done=False):
        self.config = config
        self.time = time
        self.r_net = r_net
        self.env = env
        self.action_size = config.action_size
        self.variables = variables
        self.state_det = self.get_states_det(time)
        inp = self.get_input(state_env, self.state_det)
        # self.next_states = s_net.predict(inp)
        rewards = r_net.predict(inp)
        self.rewards = np.squeeze(rewards)
        self.child_priors = self.calc_child_priors()
        if all(self.rewards <= 0):
            self.rewards = np.random.random(self.action_size)
        self.reward = reward
        self.gamma = 0.99
        self.terminal = done

    @property
    def value_estimate(self):
        return np.amax(self.rewards) * self.gamma

    def calc_child_priors(self):
        idx = (-self.rewards).argsort()[:20]
        matrix = np.zeros(self.action_size) + 1e-5
        matrix[idx] = 1
        clipped_rew = np.clip(self.rewards, 1e-5, 999)
        r = (matrix * clipped_rew) ** 0.2
        return r/sum(r)

    def get_input(self, state, det_state):
        states = np.tile(state, (self.action_size, 1))
        det_states = np.tile(det_state, (self.action_size, 1))
        actions = self.config.actions
        return states, det_states, actions

    def get_states_det(self, time):
        state_el = self.env.el_loadprofile.iat[time-1]/10
        state_th = self.env.th_loadprofile.iat[time-1, 0]/1.26
        state_bev = self.env.bev.at_home.iat[time-1, 0]
        state_pv = self.env.pv.timeseries.iat[time-1, 0] * 4
        state_temp = (self.env.temperature.iat[time-1, 0] + 10.4) / 47

        states_det = [state_el, state_th, state_bev, state_pv, state_temp]
        return states_det

    def transition(self, action):
        state, r, d, variables = self.env.step_forecast(action, self.variables)
        if r == -1:
            self.child_priors[action] = 1e-9
        return StateNode(state, self.env, self.r_net, variables["time"], self.config, variables, r, done=d)


class DummyNode(object):
    def __init__(self):
        self.dummy = True
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.value_estimates = np.zeros([8])
        self.number_visits = 0
        self.total_value = 0
