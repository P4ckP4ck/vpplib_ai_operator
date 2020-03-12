import numpy as np
import os

class PrioritizedExperienceReplay:
    def __init__(self, config, net):
        self.config = config
        self.net = net
        self.dict_idx = config.idx
        self.memory = {name: [] for name in config.idx}
        self.batch_size = config.batch_size
        self.beta = config.beta
        self.beta_anneal = config.beta_anneal
        self.max_beta = config.max_beta
        self.alpha = config.alpha
        self.load()

    def __call__(self):
        return self.memory

    def __len__(self):
        return len(self.memory["env_states"])

    @property
    def p_values(self):
        return self.memory["p_values"] + 1e-9

    @property
    def p_values_state(self):
        return self.memory["p_values_state"] + 1e-9

    @property
    def p_values_reward(self):
        return self.memory["p_values_reward"] + 1e-9

    @property
    def probabilities(self):
        p_i = self.p_values ** self.alpha
        p_sum = np.sum(self.p_values ** self.alpha)
        return np.array(p_i / p_sum, dtype=float)

    @property
    def current_batch_size(self):
        return min(self.batch_size, len(self))

    def load(self):
        for file in os.listdir("./samples/"):
            t_dict = np.load(f"./samples/{file}", allow_pickle=True).item()
            for name in self.dict_idx:
                self.memory[name] += t_dict[name]
        for name in self.dict_idx:
            self.memory[name] = np.array(self.memory[name])
        self.memory["p_values"] = np.zeros(len(self))

    def prioritize(self):
        indices = np.random.choice(range(len(self)),
                                   size=self.current_batch_size,
                                   p=self.probabilities,
                                   replace=False)
        return indices

    def update_beta(self):
        if self.beta < self.max_beta:
            self.beta *= self.beta_anneal
        else:
            self.beta = self.max_beta

    def update_p_values(self, idx, loss):
        self.memory["p_values"][idx] = loss

    def update_all_p_values(self):
        actions = np.array([self.config.asg.get_action(act)[0] for act in self.memory["actions"]])
        next_states, rewards = self.net.predict([self.memory["env_states"], self.memory["det_states"], actions])
        state_loss = np.sum(np.abs(self.memory["env_next"] - next_states), axis=1)
        rew_loss = np.abs(self.memory["rewards"] - np.squeeze(rewards))
        self.memory["p_values_state"] = state_loss
        self.memory["p_values_reward"] = rew_loss
        self.memory["p_values"] = rew_loss + state_loss

    def get_batch(self):
        self.update_all_p_values()
        idx = self.prioritize()
        batch = {name: self.memory[name][idx] for name in self.dict_idx}
        return batch, idx

    def get_is_weights(self, idx):
        p_values_state = self.memory["p_values_state"][idx]
        is_weights_state = ((1 / len(self)) * (1 / p_values_state)) ** self.beta
        p_values_reward = self.memory["p_values_reward"][idx]
        is_weights_reward = ((1 / len(self)) * (1 / p_values_reward)) ** self.beta
        return [is_weights_state, is_weights_reward]

    def get_sample_weights(self, idx):
        sample_weights = self.get_is_weights(idx)
        # self.update_beta()
        return sample_weights

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = abs(error) <= clip_delta

        squared_loss = 0.5 * error ** 2
        quadratic_loss = 0.5 * clip_delta ** 2 + clip_delta * (abs(error) - clip_delta)
        return abs(np.mean(np.where(cond, squared_loss, quadratic_loss)))

    def append(self, values):
        if len(self) == 0:
            self.memory = np.array([values])
        else:
            self.memory = np.vstack([self.memory, values])

        if len(self) > self.config.memory_size:
            self.memory = self.memory[-self.config.memory_size:]

    def collect_training_data(self):
        batch, idx = self.get_batch()
        sample_weights = self.get_sample_weights(idx)
        return batch, sample_weights

if __name__=="__main__":
    import helper
    from evaluator import evaluator_net
    net = evaluator_net()
    per = PrioritizedExperienceReplay(helper.Config(), net)
    print(np.array(per.memory["env_states"]).shape, per.memory["p_values"])
    a, b = per.get_batch()
    asg = helper.ActionSpaceGenerator()
    actions = np.array([asg.get_action(act)[0] for act in a["actions"]])
    det = []
    for d in a["det_states"]:
        det.append(d)
    t_next_states, t_rewards = net.predict([a["env_states"], a["det_states"], actions])