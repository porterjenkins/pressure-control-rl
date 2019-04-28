from simulator import Simulator
import json

MODELS = ['linear','tabular']

STATE_SIZE = {'linear':8, 'tabular': 100,
              'mlp': 8, 'lstm': 8}

for m_name in MODELS:
    print("starting " + m_name)
    sim = Simulator(controller=m_name, totalsteps=300000, target=300, state_size=STATE_SIZE[m_name], persist=True,
                    target_net_update=10)
    # sim.controller.load_model('models/q-table.p')
    # train
    output = sim.main(n_episodes=5)

    with open("output/{}.json".format(m_name), 'w') as f:
        json.dump(output, f)

