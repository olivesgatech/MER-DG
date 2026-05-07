"""
Feature Extraction Script for EPIC-rgb-flow-audio
Loads trained models and extracts backbone features before projection/classification heads
Saves features as .pt files for downstream tasks
"""

from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_EPIC_MERDG import EPICDOMAIN
from dataloader_EPIC_MERDG_lmdb import EPICDOMAIN_LMDB
from datetime import datetime


def extract_features(model, model_flow, audio_model, audio_cls_model, dataloader, args, split_name):
    """
    Extract features from the backbone networks before projection/classification heads
    
    Returns:
        features_dict: Dictionary containing:
            - video_features: List of video backbone features (if use_video)
            - flow_features: List of flow backbone features (if use_flow)
            - audio_features: List of audio backbone features (if use_audio)
            - labels: List of action labels
            - domain_labels: List of domain labels
            - video_ids: List of video identifiers
    """
    features_dict = {
        'video_features': [],
        'flow_features': [],
        'audio_features': [],
        'labels': [],
        'domain_labels': [],
        'video_ids': []
    }
    
    print(f"Extracting features for {split_name} split...")
    
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, (clip, flow, spectrogram, labels, domain_labels) in enumerate(dataloader):
                batch_size = labels.size(0)
                
                # Extract Video Features (from backbone, before projection heads)
                if args.use_video:
                    clip_tensor = clip['imgs'].cuda().squeeze(1)
                    # Get raw backbone features
                    x_slow, x_fast = model.module.backbone.get_feature(clip_tensor)
                    # Get the processed features (after get_predict, which returns tuple)
                    v_feat = model.module.backbone.get_predict((x_slow, x_fast))
                    # Pass through cls_head to get the embedding (before final classification)
                    # cls_head returns (prediction, embedding)
                    _, v_emd = model.module.cls_head(v_feat)
                    # Store as numpy to save memory
                    features_dict['video_features'].append(v_emd.cpu().numpy())
                
                # Extract Flow Features (from backbone, before projection heads)
                if args.use_flow:
                    flow_tensor = flow['imgs'].cuda().squeeze(1)
                    # Get raw backbone features
                    f_feat = model_flow.module.backbone.get_feature(flow_tensor)
                    # Get the processed features (after get_predict)
                    f_feat = model_flow.module.backbone.get_predict(f_feat)
                    # Pass through cls_head to get the embedding
                    _, f_emd = model_flow.module.cls_head(f_feat)
                    features_dict['flow_features'].append(f_emd.cpu().numpy())
                
                # Extract Audio Features (from audio backbone, before projection heads)
                if args.use_audio:
                    spectrogram_tensor = spectrogram.unsqueeze(1).cuda()
                    # Get audio features from AVENet backbone
                    _, audio_feat, _ = audio_model(spectrogram_tensor)
                    # Pass through audio_cls_model to get the embedding
                    _, audio_emd = audio_cls_model(audio_feat)
                    features_dict['audio_features'].append(audio_emd.cpu().numpy())
                
                # Store labels and metadata
                features_dict['labels'].append(labels.numpy())
                features_dict['domain_labels'].append(domain_labels.numpy())
                
                # Store video IDs if available (from dataset)
                if hasattr(dataloader.dataset, 'samples'):
                    video_ids = [dataloader.dataset.samples[idx][0] for idx in range(i*batch_size, min((i+1)*batch_size, len(dataloader.dataset)))]
                    features_dict['video_ids'].extend(video_ids)
                
                pbar.set_postfix_str(f"Processed {i+1}/{len(dataloader)} batches")
                pbar.update()
    
    # Concatenate all batches
    if args.use_video and len(features_dict['video_features']) > 0:
        features_dict['video_features'] = np.concatenate(features_dict['video_features'], axis=0)
        print(f"Video features shape: {features_dict['video_features'].shape}")
    else:
        features_dict['video_features'] = None
        
    if args.use_flow and len(features_dict['flow_features']) > 0:
        features_dict['flow_features'] = np.concatenate(features_dict['flow_features'], axis=0)
        print(f"Flow features shape: {features_dict['flow_features'].shape}")
    else:
        features_dict['flow_features'] = None
        
    if args.use_audio and len(features_dict['audio_features']) > 0:
        features_dict['audio_features'] = np.concatenate(features_dict['audio_features'], axis=0)
        print(f"Audio features shape: {features_dict['audio_features'].shape}")
    else:
        features_dict['audio_features'] = None
    
    features_dict['labels'] = np.concatenate(features_dict['labels'], axis=0)
    features_dict['domain_labels'] = np.concatenate(features_dict['domain_labels'], axis=0)
    
    print(f"Total samples: {len(features_dict['labels'])}")
    print(f"Labels shape: {features_dict['labels'].shape}")
    print(f"Domain labels shape: {features_dict['domain_labels'].shape}")
    
    return features_dict


def load_models(args, device):
    """
    Load the trained models from checkpoint
    """
    print("Loading models...")
    
    # Initialize configurations
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'
    
    model = None
    model_flow = None
    audio_model = None
    audio_cls_model = None
    cfg = None
    cfg_flow = None
    
    # Load Video Model
    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, 8).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)
        model.eval()
        print("Video model loaded")
    
    # Load Flow Model
    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, 8).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)
        model_flow.eval()
        print("Flow model loaded")
    
    # Load Audio Model and Audio Classification Model
    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()
        
        # Load audio classification model
        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, 8)
        audio_cls_model = audio_cls_model.cuda()
        audio_cls_model.eval()
        print("Audio model loaded")
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading trained weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        
        if args.use_video and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded video model weights")
            
        if args.use_flow and 'model_flow_state_dict' in checkpoint:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            print("Loaded flow model weights")
            
        if args.use_audio and 'audio_model_state_dict' in checkpoint:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            print("Loaded audio model weights")
            
        if args.use_audio and 'audio_cls_model_state_dict' in checkpoint:
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
            print("Loaded audio classification model weights")
        
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best Val Acc: {checkpoint.get('BestAcc', 'unknown')}")
        print(f"Best Test Acc: {checkpoint.get('BestTestAcc', 'unknown')}")
    elif args.checkpoint_path:
        print(f"Warning: Checkpoint path provided but file not found: {args.checkpoint_path}")
        print("Using pretrained weights only (no fine-tuned weights)")
    else:
        print("No checkpoint provided, using pretrained weights only")
    
    return model, model_flow, audio_model, audio_cls_model, cfg, cfg_flow


def main():
    parser = argparse.ArgumentParser(description='Extract features from trained models')
    
    # Model selection
    parser.add_argument('--use_video', action='store_true', help='Extract video features')
    parser.add_argument('--use_audio', action='store_true', help='Extract audio features')
    parser.add_argument('--use_flow', action='store_true', help='Extract flow features')
    
    # Checkpoint
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained model checkpoint (.pt file)')
    
    # Data configuration
    parser.add_argument('-d', '--domain', nargs='+', required=True,
                        help='Domain(s) to extract features from')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to extract features from')
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='Path to EPIC-KITCHENS dataset')
    
    # LMDB configuration
    parser.add_argument('--use_lmdb', action='store_true', default=False,
                        help='Use LMDB for faster frame loading')
    parser.add_argument('--lmdb_path', type=str, default='/home/yavuz/data/EPIC/frames_rgb_flow_lmdb/',
                        help='Path to LMDB files')
    
    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='extracted_features/',
                        help='Directory to save extracted features')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Custom name for output file (optional)')
    parser.add_argument('--use_checkpoint_folder', action='store_true',
                        help='Create a subfolder named after the checkpoint experiment')
    
    args = parser.parse_args()
    
    # Validate at least one modality is selected
    if not (args.use_video or args.use_audio or args.use_flow):
        raise ValueError("At least one modality must be selected (--use_video, --use_audio, or --use_flow)")
    
    # Determine output directory structure
    if args.use_checkpoint_folder and args.checkpoint_path:
        # Extract experiment name from checkpoint path
        # Expected format: models/EPIC_video_audio_src['D1', 'D2']_tgt['D3']_seed0_timestamp/checkpoint.pt
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        exp_name = os.path.basename(checkpoint_dir)
        
        # If checkpoint is directly in models/ folder (old format), use checkpoint filename
        if exp_name == 'models' or not exp_name:
            checkpoint_filename = os.path.basename(args.checkpoint_path)
            exp_name = os.path.splitext(checkpoint_filename)[0]  # Remove .pt extension
        
        # Create experiment-specific features folder
        features_output_dir = os.path.join(args.output_dir, exp_name)
        print(f"Using checkpoint-specific folder: {exp_name}")
    else:
        features_output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(features_output_dir, exist_ok=True)
    print(f"Features will be saved to: {features_output_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    model, model_flow, audio_model, audio_cls_model, cfg, cfg_flow = load_models(args, device)
    
    # Choose dataloader based on LMDB flag
    if args.use_lmdb:
        print(f"Using LMDB dataloader from: {args.lmdb_path}")
        DataLoader = EPICDOMAIN_LMDB
        extra_kwargs = {'lmdb_path': args.lmdb_path, 'use_lmdb': True}
    else:
        print("Using file-based dataloader")
        DataLoader = EPICDOMAIN
        extra_kwargs = {}
    
    # Create dataset and dataloader
    print(f"Loading {args.split} dataset for domain(s): {args.domain}")
    dataset = DataLoader(
        split=args.split,
        domain=args.domain,
        cfg=cfg,
        cfg_flow=cfg_flow,
        datapath=args.datapath,
        use_video=args.use_video,
        use_flow=args.use_flow,
        use_audio=args.use_audio,
        **extra_kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Extract features
    features_dict = extract_features(model, model_flow, audio_model, audio_cls_model, dataloader, args, args.split)
    
    # Generate output filename
    if args.output_name:
        output_filename = args.output_name
    else:
        modalities = []
        if args.use_video:
            modalities.append('video')
        if args.use_flow:
            modalities.append('flow')
        if args.use_audio:
            modalities.append('audio')
        modality_str = '_'.join(modalities)
        domain_str = '_'.join(args.domain)
        output_filename = f"features_{modality_str}_{domain_str}_{args.split}.pt"
    
    output_path = os.path.join(features_output_dir, output_filename)
    
    # Save features
    print(f"\nSaving features to: {output_path}")
    torch.save(features_dict, output_path)
    print("Features saved successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Output file: {output_path}")
    if args.checkpoint_path:
        print(f"Source checkpoint: {args.checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Domain(s): {args.domain}")
    print(f"Total samples: {len(features_dict['labels'])}")
    print(f"Modalities extracted:")
    if args.use_video:
        print(f"  - Video: {features_dict['video_features'].shape if features_dict['video_features'] is not None else 'N/A'}")
    if args.use_flow:
        print(f"  - Flow: {features_dict['flow_features'].shape if features_dict['flow_features'] is not None else 'N/A'}")
    if args.use_audio:
        print(f"  - Audio: {features_dict['audio_features'].shape if features_dict['audio_features'] is not None else 'N/A'}")
    print("="*50)
    
    # Save metadata file
    metadata_path = output_path.replace('.pt', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Feature Extraction Metadata\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Output folder: {features_output_dir}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Domain(s): {args.domain}\n")
        f.write(f"Total samples: {len(features_dict['labels'])}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Use LMDB: {args.use_lmdb}\n")
        f.write(f"\nFeature shapes:\n")
        if args.use_video and features_dict['video_features'] is not None:
            f.write(f"  Video: {features_dict['video_features'].shape}\n")
        if args.use_flow and features_dict['flow_features'] is not None:
            f.write(f"  Flow: {features_dict['flow_features'].shape}\n")
        if args.use_audio and features_dict['audio_features'] is not None:
            f.write(f"  Audio: {features_dict['audio_features'].shape}\n")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
