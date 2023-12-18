import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#change here how many latent variables you want to have
parser.add_argument('--z',         type=int, default=2,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
#set the training flag to 1 to train the encoder and decoder. Setting it to zero will trigger the evaluation procedure only.
parser.add_argument('--train',     type=int, default=0,     help="Flag for training")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
vae = VAE(z_dim=args.z, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
    train(model=vae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)


else:
    ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
    
    #sample from distribution p(x)
    ut.visualize_samples(vae,200,(10,20),(20,7))

    #visualize mappings to 2D latent space
    if args.z == 2:
        tl_list = list(train_loader)
        x = torch.concatenate([tl_list[i][0] for i in range(len(tl_list))])
        y = torch.concatenate([tl_list[i][1] for i in range(len(tl_list))])
        ut.visualize_2d_mappings(vae,x,y)

