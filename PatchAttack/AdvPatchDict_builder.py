import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as Models
import torch.optim as optim
from torch.utils.data import DataLoader
import PatchAttack.utils as utils
from PatchAttack.TextureDict_extractor import vgg19_extractor as gen_kit
import kornia
import time
from PatchAttack.PatchAttack_config import PA_cfg

torch_cuda = 0

class custom_data_agent():

    def __init__(self, dataset):
        self.train_dataset = dataset

    def update_loaders(self, batch_size):
        
        self.batch_size = batch_size

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )

        print('Your dataloader have been updated with batch size {}'
              .format(self.batch_size))


def make_training_batch(input_tensor, patch, patch_mask):

    # determine patch size
    H, W = PA_cfg.image_shape[-2:]
    PATCH_SIZE = int(np.floor(np.sqrt((H*W*PA_cfg.percentage))))

    translate_space = [H-PATCH_SIZE+1, W-PATCH_SIZE+1]
    bs = input_tensor.size(0)

    training_batch = []
    for b in range(bs):
        
        # random translation
        u_t = np.random.randint(low=0, high=translate_space[0])
        v_t = np.random.randint(low=0, high=translate_space[1])
        # random scaling and rotation
        scale = np.random.rand() * (PA_cfg.scale_max - PA_cfg.scale_min) + PA_cfg.scale_min
        scale = torch.Tensor([scale])
        angle = np.random.rand() * (PA_cfg.rotate_max - PA_cfg.rotate_min) + PA_cfg.rotate_min
        angle = torch.Tensor([angle])
        center = torch.Tensor([u_t+PATCH_SIZE/2, v_t+PATCH_SIZE/2]).unsqueeze(0)
        rotation_m = kornia.get_rotation_matrix2d(center, angle, scale)

        # warp three tensors
        temp_mask = patch_mask.unsqueeze(0)
        temp_input = input_tensor[b].unsqueeze(0)
        temp_patch = patch.unsqueeze(0)

        temp_mask = kornia.translate(temp_mask.float(), translation=torch.Tensor([u_t, v_t]).unsqueeze(0))
        temp_patch = kornia.translate(temp_patch.float(), translation=torch.Tensor([u_t, v_t]).unsqueeze(0))

        mask_warpped = kornia.warp_affine(temp_mask.float(), rotation_m, temp_mask.size()[-2:])
        patch_warpped = kornia.warp_affine(temp_patch.float(), rotation_m, temp_patch.size()[-2:])

        # overlay
        overlay = temp_input * (1 - mask_warpped) + patch_warpped * mask_warpped
        
        training_batch.append(overlay)
    
    training_batch = torch.cat(training_batch, dim=0)
    return training_batch


def build(model, t_labels, DA):

    # determine patch size
    H, W = PA_cfg.image_shape[-2:]
    PATCH_SIZE = int(np.floor(np.sqrt((H*W*PA_cfg.percentage))))

    # make loaders
    DA.update_loaders(PA_cfg.batch_size)

    # initilize patch and patch mask
    patch = torch.zeros(PA_cfg.image_shape)
    patch_mask = torch.zeros(1, H, W)
    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            patch_mask[:, i, j] = 1.

    # make patch optimizable
    patch = patch.requires_grad_()
    optimizer = optim.SGD(params=[patch], lr=PA_cfg.AP_lr)

    # criterion
    criterion = nn.CrossEntropyLoss().cuda(torch_cuda)

    # build dictionary of adv patches
    for AP_index in range(len(PA_cfg.AdvPatch_dirs)):

        t_label = t_labels[AP_index]

        # check existence
        if os.path.exists(
            os.path.join(
                PA_cfg.AdvPatch_dirs[AP_index],
                'patch_with_mask.pt',
            )
        ):
            print('patch of t_label{} is already generated!'.format(t_label))

        else:
            # train the patch
            target_tensor = torch.ones(PA_cfg.batch_size).long().cuda(torch_cuda)*t_label
            time_start = time.time()

            for i, (input_tensor, label_tensor) in enumerate(DA.train_loader):
                # make one batch of patches
                training_batch = make_training_batch(input_tensor, patch, patch_mask)
                training_batch, label_tensor = training_batch.cuda(torch_cuda), label_tensor.cuda(torch_cuda)

                output = model(training_batch)
                loss = criterion(output, target_tensor)

                loss.backward()

                target_acc = utils.accuracy(output.data, target_tensor, topk=(1,))

                optimizer.step()
                optimizer.zero_grad()

                # clamp the patch
                gen_kit.normalize(patch)

                print('Target: {} | iter [{}] | loss: {:.8f} | Target acc: {:.4f}'.format(t_label, i, loss.item(), target_acc[0].item()))

                if i > PA_cfg.iterations:
                    break

            # save patch
            if not os.path.exists(PA_cfg.AdvPatch_dirs[AP_index]):
                os.makedirs(PA_cfg.AdvPatch_dirs[AP_index])
            torch.save((patch.detach(), patch_mask.clone()), os.path.join(PA_cfg.AdvPatch_dirs[AP_index], 'patch_with_mask.pt'))

            time_end = time.time()
            print('t_label {} finished | time used: {}'.format(t_label, time_end - time_start))

    





