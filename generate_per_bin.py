import torch
import os
import argparse
from tqdm import tqdm
from mesh2tex import data
from mesh2tex import config_per_bin
from mesh2tex.checkpoints import CheckpointIO

# Get arguments and Config
parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()
cfg = config_per_bin.load_config(args.config, 'configs/default.yaml')

# Define device
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Read config
out_dir = cfg['training']['out_dir']
vis_dir = cfg['test']['vis_dir']
split = cfg['test']['dataset_split']
if split != 'test_vis' and split != 'test_eval':
    print('Are you sure not using test data?')
batch_size = cfg['generation']['batch_size']
gen_mode = cfg['test']['generation_mode']
model_url = None # cfg['model']['model_url'] # CHANGED BY ERIK SANDSTROEM

# Run the below code 24 times, one time for each bin
for i in range(24):
    binIdx = i
    #print(i)
    if binIdx<= 17: 
        binIdx = binIdx + 6 
        #Relating 0th bin index to 6th image index.  
    else:
        binIdx=i-18  

    if i is not 26: 
        # Dataset
        dataset = config_per_bin.get_dataset(split, cfg, binIdx, input_sampling=False)  

        if cfg['data']['shapes_multiclass']: # This is false
            datasets = dataset.datasets_classes
        else:
            datasets = [dataset]

        # Load Model
        models = config_per_bin.get_models(cfg, device=device, dataset=dataset)
        model_g = models['generator']
        checkpoint_io = CheckpointIO(out_dir, model_g=model_g)
        if model_url is None:
            checkpoint_io.load(cfg['test']['model_file'])
        else:
            checkpoint_io.load(cfg['model']['model_url'])

        # Assign Generator
        generator = config_per_bin.get_generator(model_g, cfg, device)

        # data iteration loop
        for i_ds, ds in enumerate(datasets):
            # there is only one dataset so this loop runs only one time
            ds_id = ds.metadata.get('id', str(i_ds))
            ds_name = ds.metadata.get('name', 'n/a')
            
            if cfg['data']['shapes_multiclass']:
                out_dir = os.path.join(vis_dir, ds_id)
            else:
                out_dir = vis_dir + '_bin_' + str(binIdx)
                
            test_loader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=12, shuffle=False, 
                    collate_fn=data.collate_remove_none)

            batch_counter = 0

            def get_batch_size(batch):
                batch_size = next(iter(batch.values())).shape[0]
                return batch_size

            for batch in tqdm(test_loader):
                offset_batch = batch_size * batch_counter
                if batch == None:
                    continue
                model_names = [
                    ds.get_model(i) for i in batch['idx']
                ]
                #print('model_names: ', model_names)

                if gen_mode == 'interpolate':
                    out = generator.generate_images_4eval_vae_interpol(batch,
                                                                       out_dir,
                                                                       model_names)
                elif gen_mode == 'vae':
                    out = generator.generate_images_4eval_vae(batch,
                                                              out_dir,
                                                              model_names)
                elif gen_mode == 'gan':
                    out = generator.generate_images_4eval_gan(batch,
                                                              out_dir,
                                                              model_names)
                elif gen_mode == 'interpolate_rotation':
                    out = generator.generate_images_4eval_vae_inter_rot(batch,
                                                                        out_dir,
                                                                        model_names)
                elif gen_mode == 'HD':
                    generator.generate_images_4eval_condi_hd(batch,
                                                             out_dir,
                                                             model_names)

                elif gen_mode == 'SD':
                    generator.generate_images_4eval_condi(batch,
                                                          out_dir,
                                                          model_names)

                elif gen_mode == 'grid':
                    out = generator.generate_grid(batch,
                                                  out_dir,
                                                  model_names)

                elif gen_mode == 'test':
                    out = generator.generate_images_occnet(batch,
                                                           out_dir,
                                                           model_names)
                else:
                    print('Modes: HD, grid, interpolate, interpolate_rotation, test')

                batch_counter += 1
