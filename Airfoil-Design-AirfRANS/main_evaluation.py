import yaml, json
import torch
import utils.metrics as metrics
from dataset.dataset import Dataset
import os.path as osp
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--my_path', default='/data/path', type=str,
                    help='Path to the AirfRANS root or directly to its Dataset folder')
parser.add_argument('--save_path', default='./', type=str,
                    help='Root containing metrics/<task>/<model>/<model> unless --checkpoint_path is given')
parser.add_argument('--checkpoint_path', default=None, type=str,
                    help='Optional explicit checkpoint path. Accepts a model, a list of models, or latest_checkpoint.pth')
parser.add_argument('-t', '--task', default='full', type=str,
                    choices=['full', 'scarce', 'reynolds', 'aoa'])
parser.add_argument('--model', default='Transolver', type=str)
parser.add_argument('--results_path', default=None, type=str,
                    help='Writable output directory. Defaults to <save_path>/scores/<task>')
parser.add_argument('--n_test', default=3, type=int,
                    help='Number of test airfoils to save detailed VTK/curve outputs for')
args = parser.parse_args()


def resolve_dataset_dir(path):
    if osp.exists(osp.join(path, 'manifest.json')):
        return path

    dataset_dir = osp.join(path, 'Dataset')
    if osp.exists(osp.join(dataset_dir, 'manifest.json')):
        return dataset_dir

    raise FileNotFoundError(
        f'Could not find manifest.json in "{path}" or "{dataset_dir}". '
        'Pass --my_path as either the AirfRANS root or the Dataset folder.'
    )


def load_model_stack(path, device):
    loaded = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and 'model' in loaded:
        loaded = loaded['model']
    if not isinstance(loaded, list):
        loaded = [loaded]
    return [model.to(device) for model in loaded]


use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

data_dir = resolve_dataset_dir(args.my_path)
ckpt_root_dir = args.save_path

tasks = [args.task]

for task in tasks:
    print('Generating results for task ' + task + '...')
    s = task + '_test' if task != 'scarce' else 'full_test'
    s_train = task + '_train'

    with open(osp.join(data_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    manifest_train = manifest[s_train]
    n = int(.1 * len(manifest_train))
    train_dataset = manifest_train[:-n]

    _, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=data_dir)

    # Compute the scores on the test set

    model_names = [args.model]
    models = []
    hparams = []

    for model in model_names:
        model_path = args.checkpoint_path or osp.join(ckpt_root_dir, 'metrics', task, model, model)
        mod = load_model_stack(model_path, device)
        print(mod)
        models.append(mod)

        with open('params.yaml', 'r') as f:
            hparam = yaml.safe_load(f)[model]
            hparams.append(hparam)

    results_dir = args.results_path or osp.join(ckpt_root_dir, 'scores', task)
    coefs = metrics.Results_test(device, models, hparams, coef_norm, data_dir, results_dir, n_test=args.n_test, criterion='MSE',
                                 s=s)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.

    np.save(osp.join(results_dir, 'true_coefs'), coefs[0])
    np.save(osp.join(results_dir, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join(results_dir, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(results_dir, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(results_dir, 'surf_coefs_' + str(n)), file)
    np.save(osp.join(results_dir, 'true_bls'), coefs[5])
    np.save(osp.join(results_dir, 'bls'), coefs[6])
