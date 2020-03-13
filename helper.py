import numpy as np
import time


class Config:
    def __init__(self):
        # PER configs
        self.idx = ["env_states", "det_states", "actions", "env_next", "rewards"]
        self.batch_size = 500000
        self.memory_size = 50000
        self.beta = 0.7
        self.beta_anneal = 1.00005
        self.max_beta = 0.7
        self.alpha = 0.6

        # main configs
        self.episodes = 10
        self.epochs = 1
        self.training_epochs = 100
        self.training_batch_size = 5000
        self.simulations = 500
        self.environment_steps = 96
        self.asg = ActionSpaceGenerator()
        self.action_size = self.asg.action_size
        self.actions = np.array(list(self.asg.action_dict.values()))

        # network configs
        self.units = 16
        self.obs_env = 3
        self.obs_det = 5
        self.learning_rate = 0.0001
        self.lr_decay = self.learning_rate/self.training_epochs


class SampleHistory:
    def __init__(self, obs=16, actions=8, force_reload=False):
        self.obs = obs
        self.actions = actions
        self.force_reload = force_reload
        self.history = self.load()
        self.states = self.history["states"]
        self.visits = self.history["visits"]
        self.values = self.history["values"]

    def update(self, history):
        self.states = np.vstack([self.states, history["states"]])
        self.visits = np.vstack([self.visits, history["visits"]])
        self.values = np.hstack([self.values, history["values"]])
        self.save()

    def flush(self):
        self.__init__(self.obs, self.actions, force_reload=True)
        return self

    def load(self):
        try:
            history = np.load("./checkpoints/sample_history.npy", allow_pickle=True).item()
        except:
            history = {"states": np.zeros(self.obs), "visits": np.zeros(self.actions), "values": np.zeros(1)}
        if self.force_reload:
            history = {"states": np.zeros(self.obs), "visits": np.zeros(self.actions), "values": np.zeros(1)}
        return history

    def save(self):
        history = {"states": self.states, "visits": self.visits, "values": self.values}
        np.save("./checkpoints/sample_history.npy", history)


class MinMaxStats:
    def __init__(self, init=False):
        self.min_max = self.load(init)
        self.min = self.min_max["min"]
        self.max = self.min_max["max"]
        self.high_score = self.min_max["high_score"]

    def update(self, values):
        old_min, old_max = self.min, self.max
        self.min = min(min(values), self.min)
        self.max = max(max(values), self.max)
        if self.min != old_min or self.max != old_max:
            self.save()
        return self

    def load(self, init):
        try:
            min_max = np.load("./checkpoints/min_max_stats.npy", allow_pickle=True).item()
        except:
            min_max = {"min": -0.1, "max": 0.1, "high_score": -9999}
        if init:
            min_max = {"min": -0.1, "max": 0.1, "high_score": -9999}
        return min_max

    def save(self):
        min_max = {"min": self.min, "max": self.max, "high_score": self.high_score}
        np.save("./checkpoints/min_max_stats.npy", min_max)

    def calc_summed_value(self, values, visits):
        sum_val = 0
        for i, v in enumerate(visits):
            if v != 0:
                sum_val += values[i]
        return sum_val

    def normalize(self, value, alt=True):
        if alt:
            return (((value - self.min) / (self.max - self.min)) * 2) - 1
        return (value - self.min) / (self.max - self.min)


class ActionSpaceGenerator:
    def __init__(self):
        self.heatpump = np.arange(0, 1.1, 0.2)
        self.bev = np.arange(0, 1.1, 0.25)
        self.el_store = np.arange(0, 1.1, 0.2)
        self.action_dict = self._create_action_dict()
        self.action_size = len(self.action_dict)

    def _create_action_dict(self):
        action_dict = {}
        i = 0
        for hp_step in self.heatpump:
            for bev_step in self.bev:
                for store_step in self.el_store:
                    action_dict[i] = [hp_step, bev_step, store_step]
                    i += 1
        return action_dict

    def get_action(self, action):
        # returns use-factor and flags
        return self.action_dict[action], np.ceil(self.action_dict[action]).astype(int)


class Generator:
    def __init__(self, config, env, net):
        self.dict_idx = config.idx
        self.config = config
        self.env = env
        self.net = net
        self.training_dict = {name: [] for name in self.dict_idx}

    def reset(self):
        self.training_dict = {name: [] for name in self.dict_idx}

    def get_det_states(self, time):
        state_el = self.env.el_loadprofile.iat[time-1]
        state_th = self.env.th_loadprofile.iat[time-1, 0]
        state_bev = self.env.bev.at_home.iat[time-1, 0]
        state_pv = self.env.pv.timeseries.iat[time-1, 0]
        state_temp = self.env.temperature.iat[time-1, 0]

        det_states = np.array([state_el, state_th, state_bev, state_pv, state_temp])

        return det_states

    def append(self, env_states, det_states, action, env_next, reward):
        values = [env_states, det_states, action, env_next, reward]
        for name, value in zip(self.dict_idx, values):
            self.training_dict[name].append(value)

    def add_exploration_noise(self, child_priors, dir_x=0.75, dir_alpha=1):
        priors = dir_x * child_priors + (1 - dir_x) * np.random.dirichlet([dir_alpha] * len(child_priors))
        return priors/sum(priors)

    def create_training_samples(self, greedy=False):
        asg = ActionSpaceGenerator()
        action_size = asg.action_size
        actions = np.arange(action_size)
        tack = time.time()
        i = 0
        while len(self.training_dict["env_states"]) < 50000:
            prior_state = self.env.reset()
            done = False
            while not done:
                det_state = self.get_det_states(self.env.time)
                inp = self.get_input(prior_state, det_state)
                _, rewards = self.net.predict(inp)
                if greedy:
                    action = np.argmax(rewards)
                else:
                    rewards = np.clip(np.squeeze(rewards), 0, 10)
                    try:
                        priors = self.add_exploration_noise(rewards**0.5/np.sum(rewards**0.5))
                        action = np.random.choice(actions, p=priors)  # np.random.randint(action_size)
                    except:
                        action = np.random.randint(180)
                state, reward, done, _ = self.env.step(action)
                self.append(prior_state, det_state, action, state, reward)
                prior_state = state
                i += 1
                if not i % 1000:
                    tick = time.time()
                    print(f"Time per sample {np.round(tick - tack, 4)/i}s")
                    print(len(self.training_dict["env_states"]))
        np.save(f"./samples/train_dict_{time.time()}.npy", self.training_dict)
        self.reset()

    def to_one_hot(self, actions):
        a_len = len(actions)
        zeros = np.zeros((a_len, self.config.action_size))
        for i, act in enumerate(actions):
            zeros[i][act] = 1
        return zeros

    def get_input(self, state, det_state):
        action_size = self.config.action_size
        states = np.tile(state, (action_size, 1))
        det_states = np.tile(det_state, (action_size, 1))
        actions = self.config.actions
        return states, det_states, actions

    def get_training_data(self):
        training_dict = np.load("train_dict.npy", allow_pickle=True).item()
        return training_dict

    def train(self, t):
        actions = np.array([self.config.asg.get_action(act)[0] for act in t["actions"]])
        self.net.fit([t["env_states"], t["det_states"], actions],
                     [t["env_next"], t["rewards"]],
                     epochs=100,
                     batch_size=5000)
        self.net.save_weights("weights.h5")