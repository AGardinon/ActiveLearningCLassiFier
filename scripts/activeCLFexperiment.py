# -------------------------------------------------- #
# Script for applying an active learning cycle
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import yaml
import argparse
from tqdm import tqdm

import activeclf as alclf
from activeclf.learning import active_learning_cycle, get_starting_batch

# --- func

# --- main

def main(argument):

    # read the config.yaml file provided
    exp_config = yaml.load(open(argument.config, 'r'), Loader=yaml.FullLoader)

    data = alclf.DataLoader(file_path=exp_config['dataset'],
                            target=exp_config['targetVariable'])
    
    # init the feature space for the active search
    data.feature_space(scaling=True)

    # - first batch, sampled random
    idxs = get_starting_batch(data=data.X, 
                              init_batch=exp_config['startingPoints'])

    # - init the functions to run the experiment
    if exp_config['kParam1'] and exp_config['kParam2']:
        print(f'Kernel initialized with values: A={exp_config['kParam1']}, B={exp_config['kParam2']}')
        kernel_function = exp_config['kParam1']*alclf.classification.RBF(exp_config['kParam2'])
    else:
        print(f'Kernel initialized with dafault values: A=1.0, B=1.0')
        kernel_function = 1.0*alclf.classification.RBF(1.0)

    # - start the ACLF experiment
    new_idxs = list()
    for cycle in tqdm(range(exp_config['Ncycles']), desc='Cycles'):

        idxs = idxs + new_idxs

        print(f'\n\n# ------------\n# --- Cycle {cycle}')
        print(f'ALpoints: {len(idxs)} / {len(data.X)}')

        print(f'Set up the Classifier and Acquisition function ..')
        classifier_func = alclf.ClassifierModel(model=exp_config['clfModel'],
                                                kernel=kernel_function,
                                                random_state=None)

        acquisition_func = alclf.DecisionFunction(mode=exp_config['acqMode'],
                                                  decimals=exp_config['entropyDecimals'],
                                                  seed=None)

        new_idxs = active_learning_cycle(
            feature_space=(data.X, data.y),
            idxs=idxs,
            new_batch=exp_config['newBatch'],
            clfModel=classifier_func,
            acquisitionFunc=acquisition_func,
            screeningSelection=exp_config['screeningSelection']
            )

    print('\n# END')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config', required=True, type=str, help='Cycle configuration (.yaml file).')
    args = parser.parse_args()
    main(argument=args)