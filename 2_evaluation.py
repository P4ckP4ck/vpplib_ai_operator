from trainer import Trainer
from reward_func import evaluator_net
from helper import Config
from ems_environment import ComplexEMS


if __name__=="__main__":
    config = Config()
    nets = evaluator_net(config)
    nets.load_weights("weights.h5")
    env = ComplexEMS()
    trainer = Trainer(config, nets, env)
    trainer.evaluate_current_iteration()
