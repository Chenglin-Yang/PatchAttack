import os
import numpy as np
from easydict import EasyDict as edict


# PatchAttack config
PA_cfg = edict() 

# config PatchAttack
def configure_PA(t_name, t_labels, 
                 target=False, area_occlu=0.035, n_occlu=1, rl_batch=500, steps=50,
                 HPA_bs=1,
                 MPA_color=False, 
                 TPA_n_agents=10,
                 ):

    # Dictionry's shared params
    PA_cfg.t_name = t_name
    PA_cfg.t_labels = t_labels

    # Texture dictionary
    PA_cfg.conv = 5
    PA_cfg.style_layer_choice = [1, 6, 11, 20, 29][:PA_cfg.conv]
    PA_cfg.style_channel_dims = [64, 128, 256, 512, 512][:PA_cfg.conv]
    PA_cfg.cam_thred = 0.8
    PA_cfg.n_clusters = 30
    PA_cfg.cls_w = 0
    PA_cfg.scale = 1
    PA_cfg.iter_num = 9999

    # AdvPatch dictionary
    PA_cfg.image_shape = (3, 224, 224)
    PA_cfg.scale_min = 0.9
    PA_cfg.scale_max = 1.1
    PA_cfg.rotate_max = 22.5
    PA_cfg.rotate_min = -22.5
    PA_cfg.batch_size = 16
    PA_cfg.percentage = 0.09
    PA_cfg.AP_lr = 10.0
    PA_cfg.iterations = 500


    # Attack's shared params
    PA_cfg.target = target  # targeted or non-targeted attack
    PA_cfg.n_occlu = n_occlu  # num of patches each agent can put on (default: 1), also used in 'HPA', 'MPA'
    PA_cfg.lr = 0.03  # learning rate for RL agent (default: 0.03)
    PA_cfg.rl_batch = rl_batch  # batch number when optimizing a RL agent (default: 500)
    PA_cfg.steps = steps  # steps to optimize each RL agent (default: 50), also used in HPA
    PA_cfg.sigma = 400  # sigam to control the area in HPA and MPA (default: 400.)
    PA_cfg.sigma_sched = []  # sigma schedule for the multiple occlusions (default: n-occlu * sigma), HPA and MPA
    if PA_cfg.sigma_sched == []:
        PA_cfg.sigma_sched = [PA_cfg.sigma]*PA_cfg.n_occlu

    # MPA
    PA_cfg.color = MPA_color  # flag to use MPA_RGB
    PA_cfg.critic = False  # actor-critic mode for each agent
    PA_cfg.dist_area = False  # use distributed-area mode
    PA_cfg.baseline_sub = True  # use baseline subtraction mode

    # TPA
    PA_cfg.n_agents = TPA_n_agents
    PA_cfg.f_noise = False  # filter the textures to be correctly classified by the model to fool, (default: False)
    PA_cfg.es_bnd = 1e-4  # early stop bound (default: 1e-4)
    PA_cfg.area_occlu = area_occlu  # occlusion area for each single patch (default: 0.04)
    PA_cfg.area_sched = []  # area schedule for the multiple agents (default: n-agents * area-occlu)
    if PA_cfg.area_sched == []:
        PA_cfg.area_sched = [PA_cfg.area_occlu] * PA_cfg.n_agents

    # HPA
    PA_cfg.HPA_bs = HPA_bs  # batch size for HPA (default: 1)
    # when bs is larger than 1, it means attacking mulitple images simultaneously. Othereise it is too slow.


    # Texture dict dirs
    texture_dirs = []
    texture_sub_dirs = []
    texture_template_dirs = []

    for t_label in PA_cfg.t_labels:
        texture_dir = os.path.join(
            PA_cfg.t_name,
            'attention-style_t-label_{}'.format(t_label)
        )
        texture_sub_dir = os.path.join(
            texture_dir,
            'conv_{}_cam-thred_{}_n-clusters_{}'.format(
                PA_cfg.conv, PA_cfg.cam_thred, PA_cfg.n_clusters
            )
        )
        texture_template_dir = os.path.join(
            texture_sub_dir,
            'cls-w_{}_scale_{}'.format(
                PA_cfg.cls_w, PA_cfg.scale,
            )
        )
        texture_dirs.append(texture_dir)
        texture_sub_dirs.append(texture_sub_dir)
        texture_template_dirs.append(texture_template_dir)
        
    PA_cfg.texture_dirs = texture_dirs
    PA_cfg.texture_sub_dirs = texture_sub_dirs
    PA_cfg.texture_template_dirs = texture_template_dirs

    # AdvPatch dict dirs
    PA_cfg.AdvPatch_dirs = []
    for t_label in PA_cfg.t_labels:
        AdvPatch_dir = os.path.join(
            PA_cfg.t_name,
            't-label_{}'.format(
                t_label
            ),
            'percentage_{}'.format(
                PA_cfg.percentage,
            ),
            'scale_{}-{}_rotate_{}-{}'.format(
                PA_cfg.scale_min, PA_cfg.scale_max,
                PA_cfg.rotate_min, PA_cfg.rotate_max,
            ),
            'LR_{}_batch_size_{}_iterations_{}'.format(
                PA_cfg.AP_lr, PA_cfg.batch_size, PA_cfg.iterations
            ),
        )
        PA_cfg.AdvPatch_dirs.append(AdvPatch_dir)

    # TPA attack dirs
    TPA_attack_dirs = []
    for agent_index in range(PA_cfg.n_agents):
        attack_dir = os.path.join(
            'target' if PA_cfg.target else 'non-target',
            os.path.join(*PA_cfg.texture_template_dirs[0].split('/')[2:]),
            'n-occlu_{}_f-noise_{}_lr_{}_rl-batch_{}_steps_{}_es-bnd_{}'
            .format(
                PA_cfg.n_occlu, PA_cfg.f_noise, 
                PA_cfg.lr, PA_cfg.rl_batch, PA_cfg.steps, PA_cfg.es_bnd,
            ),
            'area-sched_'+'-'.join(
                [str(item) for item in PA_cfg.area_sched[:agent_index+1]]
            )+\
            '_n-agents_{}'.format(agent_index+1),
        )
        TPA_attack_dirs.append(attack_dir)
    PA_cfg.TPA_attack_dirs = TPA_attack_dirs


    # MPA attack dirs
    attack_dir = os.path.join(
        'target' if PA_cfg.target else 'non-target',
        'n-occlu_{}_color_{}_lr_{}_critic_{}_rl-batch_{}_steps_{}'.format(
            PA_cfg.n_occlu, PA_cfg.color, PA_cfg.lr,
            PA_cfg.critic, PA_cfg.rl_batch, PA_cfg.steps,
        ),
        'sigma-sched_'+'-'.join(
            [str(item) for item in PA_cfg.sigma_sched]
        ),
    )
    PA_cfg.MPA_attack_dir = attack_dir


    # HPA attack dirs
    attack_dir = os.path.join(
        'target' if PA_cfg.target else 'non-target',
        'n-occlu_{}_steps_{}'
        .format(
            PA_cfg.n_occlu, PA_cfg.steps, 
        ),
        'sigma-sched_'+'-'.join(
            [str(item) for item in PA_cfg.sigma_sched]
        ),
    )
    PA_cfg.HPA_attack_dir = attack_dir


    # AP attack dirs
    temp_dir = os.path.join(*PA_cfg.AdvPatch_dirs[0].split('/')[2:])
    attack_dir = os.path.join(
        'target' if PA_cfg.target else 'non-target',
        temp_dir,
    )
    PA_cfg.AP_attack_dir = attack_dir
