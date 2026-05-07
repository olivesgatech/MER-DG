from mmaction.apis import init_recognizer
import torch
from optimizer import LARS
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_EPIC_MERDG import EPICDOMAIN

import torch.nn.functional as F
from losses import SupConLoss
import wandb
from lr_scheduler import WarmupCosineSchedule, LinearWarmupCosineDecayScheduler
from losses import mer_dg_loss
from datetime import datetime

class BalancedDomainSampler(torch.utils.data.Sampler):
    """Sampler that ensures each batch has equal samples from each domain"""
    def __init__(self, dataset, batch_size, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Group indices by domain
        self.domain_indices = {}
        for idx, sample in enumerate(dataset.samples):
            domain = sample[0].split('/')[0]  # Extract domain from path
            if domain not in self.domain_indices:
                self.domain_indices[domain] = []
            self.domain_indices[domain].append(idx)
        
        self.domains = list(self.domain_indices.keys())
        self.num_domains = len(self.domains)
        
        # Check if batch_size is divisible by number of domains
        if self.batch_size % self.num_domains != 0:
            raise ValueError(f"Batch size ({self.batch_size}) must be divisible by number of domains ({self.num_domains})")
        
        self.samples_per_domain = self.batch_size // self.num_domains
        
        # Calculate total batches
        min_domain_size = min(len(indices) for indices in self.domain_indices.values())
        self.batches_per_epoch = min_domain_size // self.samples_per_domain
        
        if not self.drop_last:
            # If not dropping last, we need to handle remainder
            max_batches = max(len(indices) // self.samples_per_domain for indices in self.domain_indices.values())
            self.batches_per_epoch = max_batches
    
    def __iter__(self):
        # Shuffle indices within each domain with reproducible seeding
        # Use epoch-dependent seeding for different shuffles across epochs
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        shuffled_domain_indices = {}
        for domain in self.domains:
            indices = self.domain_indices[domain].copy()
            indices = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
            shuffled_domain_indices[domain] = indices
        
        self.epoch += 1  # Increment epoch for next iteration
        
        # Generate balanced batches
        for batch_idx in range(self.batches_per_epoch):
            batch_indices = []
            
            for domain in self.domains:
                start_idx = batch_idx * self.samples_per_domain
                end_idx = start_idx + self.samples_per_domain
                
                domain_indices = shuffled_domain_indices[domain]
                
                if end_idx <= len(domain_indices):
                    # Normal case: enough samples
                    batch_indices.extend(domain_indices[start_idx:end_idx])
                elif not self.drop_last and start_idx < len(domain_indices):
                    # Handle remainder if not dropping last
                    available = domain_indices[start_idx:]
                    # Repeat samples to fill the quota
                    while len(available) < self.samples_per_domain:
                        available.extend(domain_indices[:min(len(domain_indices), self.samples_per_domain - len(available))])
                    batch_indices.extend(available[:self.samples_per_domain])
            
            if len(batch_indices) == self.batch_size:
                yield batch_indices
    
    def __len__(self):
        return self.batches_per_epoch


def train_one_step(clip, labels, flow, spectrogram, domain_labels):
    labels = labels.cuda()
    domain_labels = domain_labels.cuda()


    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()


    with torch.no_grad():
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)  
            v_feat = (x_slow.detach(), x_fast.detach())  
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)

    if args.use_video:
        v_feat = model.module.backbone.get_predict(v_feat)
        predict1, v_emd = model.module.cls_head(v_feat)

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

    if args.use_audio:    
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())

    modality_embeddings = []
    if args.use_video:
        modality_embeddings.append(v_emd)
    if args.use_audio:
        modality_embeddings.append(audio_emd)
    if args.use_flow:
        modality_embeddings.append(f_emd)

    if not modality_embeddings:
        raise ValueError("At least one modality must be enabled for training.")

    feat = torch.cat(modality_embeddings, dim=1) if len(modality_embeddings) > 1 else modality_embeddings[0]
    loss = 0.0
    predict = mlp_cls(feat)
    loss += args.alpha_ce * criterion(predict, labels)


   
    # Supervised Contrastive Learning
    # normalize the embeddings for contrastive learning
    modality_proj_norms = []
    if args.use_video:
        v_c_emd_proj = v_proj(v_emd)
        v_c_emd_proj_norm = F.normalize(v_c_emd_proj, dim=1)
        modality_proj_norms.append(v_c_emd_proj_norm)
    if args.use_audio:
        a_c_emd_proj = a_proj(audio_emd)
        a_c_emd_proj_norm = F.normalize(a_c_emd_proj, dim=1)
        modality_proj_norms.append(a_c_emd_proj_norm)
    if args.use_flow:
        f_c_emd_proj = f_proj(f_emd)
        f_c_emd_proj_norm = F.normalize(f_c_emd_proj, dim=1)
        modality_proj_norms.append(f_c_emd_proj_norm)

    # Only apply contrastive loss if there are multiple modalities
    if len(modality_proj_norms) > 1:
        emd_proj = torch.stack(modality_proj_norms, dim=1)
        loss_contrast = criterion_contrast(emd_proj, labels)
        loss = loss + args.alpha_contrast * loss_contrast
    elif len(modality_proj_norms) == 1 and args.alpha_contrast > 0:
        # Handle the single modality case if contrastive loss is enabled,
        # though it might not be meaningful. For stability, we can just skip it.
        # Or, if you want to apply it to the single modality against itself (not typical):
        # emd_proj = modality_proj_norms[0].unsqueeze(1)
        # loss_contrast = criterion_contrast(emd_proj, labels)
        # loss = loss + args.alpha_contrast*loss_contrast
        pass  # Skipping contrastive loss for a single modality to prevent NaN. 

    # Variance-Covariance Loss: apply per modality to remain robust for 1-3 inputs
    loss_mer = 0.0
    modalities_used = []
    if args.use_video:
        modalities_used.append(('video', v_c_emd_proj))
    if args.use_audio:
        modalities_used.append(('audio', a_c_emd_proj))
    if args.use_flow:
        modalities_used.append(('flow', f_c_emd_proj))

    if modalities_used:
        for _, modality_tensor in modalities_used:
            loss_mer += mer_dg_loss(modality_tensor, alpha_marg=args.alpha_marg, alpha_spec=args.alpha_spec)
        loss = loss + args.lambda_mer * loss_mer   
        
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(clip, labels, flow, spectrogram, domain_labels):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    domain_labels = domain_labels.cuda()
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip) 
            v_feat = (x_slow.detach(), x_fast.detach())  

            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)  
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)

        modality_embeddings = []
        if args.use_video:
            modality_embeddings.append(v_emd)
        if args.use_audio:
            modality_embeddings.append(audio_emd)
        if args.use_flow:
            modality_embeddings.append(f_emd)

        if not modality_embeddings:
            raise ValueError("At least one modality must be enabled for validation.")

        if len(modality_embeddings) > 1:
            feat = torch.cat(modality_embeddings, dim=1)
        else:
            feat = modality_embeddings[0]

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)

    return predict, loss

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)


class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = self.head(feat)
        return feat





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')

    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0,
                        help='alpha_contrast')
    parser.add_argument('--lambda_mer', type=float, default=3.0,
                        help='coefficient for entropy loss')
    parser.add_argument('--alpha_ce', type=float, default=1.0,
                        help='coefficient for cross-entropy loss')

    parser.add_argument('--resumef', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7,
                        help='explore_loss_coeff')
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0,
                        help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0,
                        help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
    
        # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='MER-DG-EPIC', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/team name')
    
    # Learning rate scheduler parameters
    parser.add_argument('--optimizer', type=str, default='lars', choices=['lars', 'adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--base_lr', type=float, default=0.3, help='Base learning rate when using LARS optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of epochs for warmup')
    parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate after decay')
    parser.add_argument('--use_lr_scheduler', type=bool, default=False, help='Use cosine lr scheduler with warmup')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')



    parser.add_argument('--alpha_marg', type=float, default=1.0,
                        help='Variance floor weight on common embeddings.')
    parser.add_argument('--alpha_spec', type=float, default=1,
                        help='Within specific-embedding decorrelation weight.')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    input_dim = 0

    cfg = None
    cfg_flow = None
    
    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, 8).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)

        v_proj = ProjectHead(input_dim=2304, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()

        input_dim = input_dim + 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, 8).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_proj = ProjectHead(input_dim=2048, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()

        input_dim = input_dim + 2048

    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, 8)
        audio_cls_model = audio_cls_model.cuda()

        a_proj = ProjectHead(input_dim=512, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()

        input_dim = input_dim + 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=8)
    mlp_cls = mlp_cls.cuda()


    base_path = args.checkpoint_dir
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    log_name = "log%s2%s"%(args.source_domain, args.target_domain)
    if args.use_video:
        log_name = log_name + '_video'
    if args.use_flow:
        log_name = log_name + '_flow'
    if args.use_audio:
        log_name = log_name + '_audio'
    log_name = log_name + args.appen
    
    # Initialize Weights & Biases first to get the run name
    wandb_run_name = None
    if args.use_wandb:
        # Create modality string for run name
        modalities = []
        if args.use_video:
            modalities.append('video')
        if args.use_flow:
            modalities.append('flow')
        if args.use_audio:
            modalities.append('audio')
        modality_str = '_'.join(modalities)
        
        # Create run name if not provided
        if args.wandb_run_name is None:
            run_name = f"EPIC_{modality_str}_src{args.source_domain}_tgt{args.target_domain}_seed{args.seed}"
            if args.appen:
                run_name += f"_{args.appen}"
        else:
            run_name = args.wandb_run_name
            
        wandb_run_name = run_name
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                'source_domain': args.source_domain,
                'target_domain': args.target_domain,
                'learning_rate': args.lr,
                'base_lr': args.base_lr if args.use_lr_scheduler else args.lr,
                'use_lr_scheduler': args.use_lr_scheduler,
                'warmup_epochs': args.warmup_epochs,
                'batch_size': args.bsz,
                'epochs': args.nepochs,
                'seed': args.seed,
                'use_video': args.use_video,
                'use_flow': args.use_flow,
                'use_audio': args.use_audio,
                'alpha_contrast': args.alpha_contrast,
                'lambda_mer': args.lambda_mer,
                'explore_loss_coeff': args.explore_loss_coeff,
                'temp': args.temp,
                'hidden_dim': args.hidden_dim,
                'out_dim': args.out_dim,
                'trans_hidden_num': args.trans_hidden_num,
                'modalities': modality_str,
                'dataset': 'EPIC-KITCHENS',
                'optimizer': args.optimizer,
            }
        )
    else:
        # Create run name even if not using wandb for consistent folder naming
        modalities = []
        if args.use_video:
            modalities.append('video')
        if args.use_flow:
            modalities.append('flow')
        if args.use_audio:
            modalities.append('audio')
        modality_str = '_'.join(modalities)
        run_name = f"EPIC_{modality_str}_src{args.source_domain}_tgt{args.target_domain}_seed{args.seed}"
        if args.appen:
            run_name += f"_{args.appen}"
        wandb_run_name = run_name

    # Create experiment folder for checkpoints
    checkpoint_folder = os.path.join(args.checkpoint_dir, wandb_run_name)
    os.makedirs(checkpoint_folder, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_folder}")
    
    log_path = os.path.join(base_path, log_name + '.csv')
    print(log_path)

    criterion = nn.CrossEntropyLoss() 
    criterion = criterion.cuda()
    batch_size = args.bsz

    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()

    params = list(mlp_cls.parameters())
    if args.use_video:
        params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(v_proj.parameters()) 
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters()) + list(f_proj.parameters())
    if args.use_audio:
        params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())
    


    # Use a higher base learning rate for LARS (typical for VICReg)
    actual_lr = args.base_lr if args.use_lr_scheduler else args.lr
    
    # LARS optimizer with adjusted parameters for better performance
    # Using parameters supported by your implementation

    if args.optimizer == 'lars':
        optim = LARS(params, 
                    lr=actual_lr, 
                    weight_decay=1e-6,  # Lower weight decay (typical for LARS)
                    momentum=0.9,       # Standard momentum
                    eta=0.001,          # Trust coefficient - controls how much we trust layer-wise adaptation
                    exclude_bias_n_norm=True)  # Exclude bias and normalization layers from adaptation
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optim = torch.optim.SGD(params, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Choose 'lars' or 'adam'.")

    # Set up learning rate scheduler - we'll initialize it after creating the train_dataset
    
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    if args.resumef:
        if args.resume_path:
            resume_file = args.resume_path
        else:
            resume_file = os.path.join(checkpoint_folder, log_name + '.pt')
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']

        if args.use_video:
            model.load_state_dict(checkpoint['model_state_dict'])
            v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            f_proj.load_state_dict(checkpoint['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
            a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])

        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    DataLoader = EPICDOMAIN
    extra_kwargs = {}

    train_dataset = DataLoader(split='train', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, 
                              datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, 
                              use_audio=args.use_audio, **extra_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create balanced domain sampler for training
    balanced_sampler = BalancedDomainSampler(train_dataset, batch_size=batch_size, drop_last=True, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=balanced_sampler, num_workers=args.num_workers,
                                                   pin_memory=(device.type == "cuda"),persistent_workers=True, prefetch_factor=1)
    
    # Now that we have the train_dataset, we can set up the learning rate scheduler if enabled
    if args.use_lr_scheduler:
        # Calculate steps per epoch for scheduler
        steps_per_epoch = len(balanced_sampler)  # Use sampler length instead of dataset
        total_steps = args.nepochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        
        # Create the scheduler
        lr_scheduler = WarmupCosineSchedule(
            optimizer=optim,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=args.min_lr
        )
        
        print(f"Learning rate scheduler set up with:")
        print(f"  - Base LR: {actual_lr}")
        print(f"  - Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")
        print(f"  - Total steps: {total_steps} ({args.nepochs} epochs)")
        print(f"  - Steps per epoch: {steps_per_epoch} (balanced sampling)")

    val_dataset = DataLoader(split='test', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, 
                             datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, 
                             use_audio=args.use_audio, **extra_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=False, drop_last=False,prefetch_factor=1)

    test_dataset = DataLoader(split='test', domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow, 
                              datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, 
                              use_audio=args.use_audio, **extra_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=args.num_workers, shuffle=False,
                                                      pin_memory=False, drop_last=False,prefetch_factor=1)
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    last_best_checkpoint_path = None
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                mlp_cls.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    a_proj.train(split == 'train')

                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, flow, spectrogram, labels, domain_labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(clip, labels, flow, spectrogram, domain_labels)
                            
                            # Step the learning rate scheduler if enabled
                            if args.use_lr_scheduler:
                                lr_scheduler.step()
                                if args.use_wandb and i % 20 == 0:  # Log every 20 steps to avoid clutter
                                    wandb.log({'learning_rate': optim.param_groups[0]['lr']})
                        else:
                            predict1, loss = validate_one_step(clip, labels, flow, spectrogram, domain_labels)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            

                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currenttestAcc >= BestTestAcc:
                            BestTestAcc = currenttestAcc
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                                
                                if args.use_video:
                                    save['v_proj_state_dict'] = v_proj.state_dict()
                                    save['model_state_dict'] = model.state_dict()
                                if args.use_flow:
                                    save['f_proj_state_dict'] = f_proj.state_dict()
                                    save['model_flow_state_dict'] = model_flow.state_dict()
                                if args.use_audio:
                                    save['a_proj_state_dict'] = a_proj.state_dict()
                                    save['audio_model_state_dict'] = audio_model.state_dict()
                                    save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()


                                new_best_checkpoint_path = os.path.join(checkpoint_folder, log_name + '_best_epoch_%s.pt'%(str(epoch_i)))
                                torch.save(save, new_best_checkpoint_path)
                                
                                if last_best_checkpoint_path is not None and os.path.exists(last_best_checkpoint_path):
                                    os.remove(last_best_checkpoint_path)
                                last_best_checkpoint_path = new_best_checkpoint_path

                        if args.save_checkpoint:
                            save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'optimizer': optim.state_dict(),
                                }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            
                            if args.use_video:
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['model_state_dict'] = model.state_dict()
                            if args.use_flow:
                                save['f_proj_state_dict'] = f_proj.state_dict()
                                save['model_flow_state_dict'] = model_flow.state_dict()
                            if args.use_audio:
                                save['a_proj_state_dict'] = a_proj.state_dict()
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()

                            torch.save(save, os.path.join(checkpoint_folder, log_name + '_checkpoint_epoch_%s.pt'%(str(epoch_i))))
                        
                    # Log metrics to wandb
                    if args.use_wandb:
                        loss_avg = total_loss / float(count)
                        acc_avg = acc / float(count)
                        
                        wandb_logs = {
                            f'{split}_loss': loss_avg,
                            f'{split}_accuracy': acc_avg,
                            'epoch': epoch_i
                        }
                        
                        # Add best metrics when available
                        if split == 'val':
                            wandb_logs['best_val_accuracy'] = BestAcc
                            wandb_logs['best_epoch'] = BestEpoch
                            wandb_logs['best_loss'] = BestLoss
                        elif split == 'test':
                            wandb_logs['best_test_accuracy'] = BestTestAcc
                            
                        wandb.log(wandb_logs)
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)
        
        # Log final results to wandb
        if args.use_wandb:
            wandb.log({
                'final_best_val_accuracy': BestAcc,
                'final_best_test_accuracy': BestTestAcc,
                'final_best_epoch': BestEpoch,
                'final_best_loss': BestLoss
            })
            wandb.finish()

    f.close()
