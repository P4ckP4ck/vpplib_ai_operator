from evaluator import evaluator_net
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
    #per = PrioritizedExperienceReplay(config, net)
    for episode in range(config.epochs):
        #batch, sample_weights = per.collect_training_data()
        #trainer.training_phase(batch, sample_weights)
        gen.create_training_samples(random=True)
        # training_dict = trainer.create_training_samples(env)
        # sample_history.update(training_dict)
        # sample_history = trainer.training_phase(sample_history)

