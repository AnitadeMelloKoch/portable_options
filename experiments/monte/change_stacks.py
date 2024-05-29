import dill
import numpy as np

filenames = [
        'resources/monte_env_states/room7/ladder/top_0.pkl',
]

rams = []

for filename in filenames:
    with open(filename, 'rb') as f:
        ram = dill.load(f)
        print(ram["position"])
        # rams.append(ram["ram"])

# for filename in filenames:
#     with open(filename, 'rb') as f:
#         ram = dill.load(f)
#         print(ram["agent_state"].shape)
#         ram["agent_state"] = ram["agent_state"].squeeze()
#         print(ram["agent_state"].shape)
#         ram["state"] = ram["state"].squeeze()
    
#     with open(filename, 'wb') as f:
#         dill.dump(ram, f)
