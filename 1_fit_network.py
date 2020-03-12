import numpy as np
import os
from helper import Config
from reward_func import evaluator_net
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

if __name__ == "__main__":
    config = Config()
    memory = {name: [] for name in config.idx}
    net = evaluator_net(config)
    try:
        net.load_weights("weights.h5")
    except:
        print("No fitting weights found. Creating new network.")
    for file in tqdm(os.listdir("./samples/")):
        t_dict = np.load(f"./samples/{file}", allow_pickle=True).item()
        for name in config.idx:
            memory[name] += t_dict[name]
    for name in config.idx:
        memory[name] = np.array(memory[name])
    batch = memory
    actions = np.array([config.asg.get_action(act)[0] for act in tqdm(batch["actions"])])

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    check_save = ModelCheckpoint('weights.h5', save_best_only=True, monitor='loss', mode='min')
    # lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')

    net.fit([batch["env_states"], batch["det_states"], actions],
            [batch["rewards"]],
            batch_size=1000,
            epochs=config.training_epochs,
            callbacks=[early_stopping, check_save])#, lr_reducer])
    print(np.squeeze(net.predict([batch["env_states"][:10], np.array(batch["det_states"])[:10], actions[:10]])).round(4))
    print(batch["rewards"][:10])