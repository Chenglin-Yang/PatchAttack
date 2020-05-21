import os
from parser import cfg
import PatchAttack.TextureDict_builder as TD_builder
from PatchAttack.PatchAttack_config import configure_PA
import PatchAttack.AdvPatchDict_builder as AP_builder 
from PatchAttack import utils
import torchvision.models as Models

torch_cuda = cfg.torch_cuda

def main():

    configure_PA(cfg.tdict_dir, cfg.t_labels)

    if cfg.dict == 'Texture':

        if cfg.t_data == 'ImageNet':

            # ImageNet data agent for texture generation
            DA = utils.data_agent(
                ImageNet_train_dir=cfg.ImageNet_train_dir,
                ImageNet_val_dir=cfg.ImageNet_val_dir,
                data_name='ImageNet',
                train_transform=utils.data_agent.process_PIL
            )
            TD_builder.build(DA, cfg.t_labels)
        
        elif cfg.t_data == 'custom':

            # custom dataset requirement:
            # attribute -- targets: list consisting of the labels (int)
            # methods -- __getitem__(): return image (torch.Tensor), label (int)

            #custom_dataset = ...
            #DA = TD_builder.custom_data_agent(custom_dataset)
            #TD_builder.build(DA, cfg.t_labels)
            assert False, 'Please see the commented requirements to build custom data agent' 

    elif cfg.dict == 'AdvPatch':

        # model
        model = getattr(Models, cfg.arch.lower()+str(cfg.depth))(
                pretrained=True
            ).cuda(torch_cuda).eval()

        if cfg.t_data == 'ImageNet':

            # ImageNet data agent for AdvPatch generation
            DA = utils.data_agent(
                ImageNet_train_dir=cfg.ImageNet_train_dir,
                ImageNet_val_dir=cfg.ImageNet_val_dir,
                data_name='ImageNet',
                train_transform=utils.data_agent.process_PIL,
            )
            AP_builder.build(model, cfg.t_labels, DA)

        elif cfg.t_data == 'custom':

            # custom dataset requirement:
            # methods -- __getitem__(): return image (torch.Tensor), label (int)

            #custom_dataset = ...
            #DA = AP_builder.custom_data_agent(custom_dataset)
            #AP_builder.build(model, cfg.t_labels, DA)
            assert False, 'Please see the commented requirements to build custom data agent' 


if __name__ == '__main__':
    main()