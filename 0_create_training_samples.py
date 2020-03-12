from reward_func import evaluator_net
from trainer import Trainer
from ems_environment import ComplexEMS
import helper as hlp
from proritized_experience_sampler import PrioritizedExperienceReplay


if __name__ == '__main__':
    config = hlp.Config()
    net = evaluator_net(config)
    sample_history = hlp.SampleHistory()
    env = ComplexEMS()
    try:
        net.load_weights("weights.h5")
    except:
        print("No fitting weights found. Creating new network.")
    trainer = Trainer(config, net, env)
    gen = hlp.Generator(config, env, net)
    for episode in range(config.epochs):
        gen.create_training_samples(random=True)


