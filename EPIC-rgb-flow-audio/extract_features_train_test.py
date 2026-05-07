"""
Extract features from both training and testing domains
This script extracts backbone embeddings for specified train/test domains and saves them for later use
"""

import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import Namespace

from extract_features import load_models
from dataloader_EPIC_MERDG import EPICDOMAIN
from dataloader_EPIC_MERDG_lmdb import EPICDOMAIN_LMDB
from dataloader_EPIC_MERDG_lmdb_optimized import EPICDOMAIN_OPTIMIZED_LMDB


def extract_embeddings_from_dataloader(
    dataloader,
    model,
    model_flow,
    audio_model,
    audio_cls_model,
    args,
    split_name,
    device
):
    """
    Extract embeddings from a dataloader for the specified modalities
    
    Returns:
        embeddings_dict: Dictionary containing:
            - video_features: numpy array of video embeddings (N, D) or None
            - flow_features: numpy array of flow embeddings (N, D) or None
            - audio_features: numpy array of audio embeddings (N, D) or None
            - labels: numpy array of action labels (N,)
            - domain_labels: numpy array of domain labels (N,)
    """
    print(f"\nExtracting embeddings for {split_name}...")
    
    video_feats = []
    flow_feats = []
    audio_feats = []
    labels_list = []
    domain_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {split_name}"):
            clip, flow, spectrogram, labels, domain_labels = batch
            
            # Extract Video Features
            if args.use_video and model is not None:
                clip_tensor = clip['imgs'].squeeze(1).to(device)
                x_slow, x_fast = model.module.backbone.get_feature(clip_tensor)
                v_feat = model.module.backbone.get_predict((x_slow, x_fast))
                _, v_embedding = model.module.cls_head(v_feat)
                video_feats.append(v_embedding.cpu().numpy())
            
            # Extract Flow Features
            if args.use_flow and model_flow is not None:
                flow_tensor = flow['imgs'].squeeze(1).to(device)
                f_feat = model_flow.module.backbone.get_feature(flow_tensor)
                f_feat = model_flow.module.backbone.get_predict(f_feat)
                _, f_embedding = model_flow.module.cls_head(f_feat)
                flow_feats.append(f_embedding.cpu().numpy())
            
            # Extract Audio Features
            if args.use_audio and audio_model is not None:
                spectrogram_tensor = spectrogram.unsqueeze(1).to(device)
                _, audio_feat, _ = audio_model(spectrogram_tensor)
                _, a_embedding = audio_cls_model(audio_feat)
                audio_feats.append(a_embedding.cpu().numpy())
            
            # Store labels
            labels_list.append(labels.numpy())
            domain_labels_list.append(domain_labels.numpy())
    
    # Concatenate all batches
    embeddings_dict = {
        'video_features': np.concatenate(video_feats, axis=0) if video_feats else None,
        'flow_features': np.concatenate(flow_feats, axis=0) if flow_feats else None,
        'audio_features': np.concatenate(audio_feats, axis=0) if audio_feats else None,
        'labels': np.concatenate(labels_list, axis=0),
        'domain_labels': np.concatenate(domain_labels_list, axis=0),
    }
    
    # Print shapes
    print(f"\n{split_name} features extracted:")
    print(f"  Total samples: {len(embeddings_dict['labels'])}")
    if embeddings_dict['video_features'] is not None:
        print(f"  Video shape: {embeddings_dict['video_features'].shape}")
    if embeddings_dict['flow_features'] is not None:
        print(f"  Flow shape: {embeddings_dict['flow_features'].shape}")
    if embeddings_dict['audio_features'] is not None:
        print(f"  Audio shape: {embeddings_dict['audio_features'].shape}")
    
    return embeddings_dict


def build_dataloader(split, domains, cfg, cfg_flow, args):
    """Build dataloader for the specified split and domains"""
    
    # Choose dataloader class
    if args.use_lmdb:
        if args.use_optimized:
            DataLoader = EPICDOMAIN_OPTIMIZED_LMDB
            extra_kwargs = {'lmdb_path': args.lmdb_path, 'use_lmdb': True}
        else:
            DataLoader = EPICDOMAIN_LMDB
            extra_kwargs = {'lmdb_path': args.lmdb_path, 'use_lmdb': True}
    else:
        DataLoader = EPICDOMAIN
        extra_kwargs = {}
    
    print(f"Loading {split} dataset for domain(s): {domains}")
    dataset = DataLoader(
        split=split,
        domain=domains,
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
        pin_memory=True,
        drop_last=False
    )
    
    print(f"  Dataset size: {len(dataset)} samples")
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Extract features from train and test domains for a list of domains.')
    
    # Model selection
    parser.add_argument('--use_video', action='store_true', help='Extract video features')
    parser.add_argument('--use_audio', action='store_true', help='Extract audio features')
    parser.add_argument('--use_flow', action='store_true', help='Extract flow features')
    
    # Checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    
    # Domain configuration
    parser.add_argument('--domains', '-d', nargs='+', required=True,
                        help='List of domains to extract features from')
    
    # Data path
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/',
                        help='Path to EPIC-KITCHENS dataset')
    
    # LMDB configuration
    parser.add_argument('--use_lmdb', action='store_true', default=False,
                        help='Use LMDB for faster frame loading')
    parser.add_argument('--use_optimized', action='store_true', default=False,
                        help='Use optimized LMDB dataloader')
    parser.add_argument('--lmdb_path', type=str, 
                        default='/home/yavuz/data/EPIC/frames_rgb_flow_lmdb/',
                        help='Path to LMDB files')
    
    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of data loading workers')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='extracted_features/',
                        help='Directory to save extracted features')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Custom name prefix for output files (optional)')
    
    args = parser.parse_args()
    
    # Validate at least one modality is selected
    if not (args.use_video or args.use_audio or args.use_flow):
        raise ValueError("At least one modality must be selected")
    
    # Validate checkpoint file exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    model, model_flow, audio_model, audio_cls_model, cfg, cfg_flow = load_models(args, device)
    
    # Set models to eval mode
    if model is not None:
        model.eval()
    if model_flow is not None:
        model_flow.eval()
    if audio_model is not None:
        audio_model.eval()
    if audio_cls_model is not None:
        audio_cls_model.eval()

    # Determine modalities for filename
    modalities = []
    if args.use_video:
        modalities.append('video')
    if args.use_flow:
        modalities.append('flow')
    if args.use_audio:
        modalities.append('audio')
    modality_str = '_'.join(modalities)

    # Loop over domains and splits
    for domain in args.domains:
        for split in ['train', 'test']:
            print("\n" + "="*60)
            print(f"PROCESSING DOMAIN: {domain}, SPLIT: {split}")
            print("="*60)

            # Build dataloader
            print("\n" + "="*30)
            print("BUILDING DATALOADER")
            print("="*30)
            
            dataloader = build_dataloader(
                split, [domain], cfg, cfg_flow, args
            )
            
            # Extract features
            print("\n" + "="*30)
            print("EXTRACTING FEATURES")
            print("="*30)
            
            embeddings = extract_embeddings_from_dataloader(
                dataloader, model, model_flow, audio_model, audio_cls_model,
                args, f"{domain} ({split})", device
            )
            
            # Generate output filenames
            if args.output_name:
                base_name = f"{args.output_name}_{modality_str}_{domain}_{split}"
            else:
                base_name = f"features_{modality_str}_{domain}_{split}"
            
            output_path = os.path.join(args.output_dir, f"{base_name}.pt")
            
            # Save features
            print("\n" + "="*30)
            print("SAVING FEATURES")
            print("="*30)
            
            print(f"Saving features to: {output_path}")
            torch.save(embeddings, output_path)
            
            # Save metadata
            metadata = {
                'checkpoint_path': args.checkpoint_path,
                'domain': domain,
                'split': split,
                'use_video': args.use_video,
                'use_flow': args.use_flow,
                'use_audio': args.use_audio,
                'samples': len(embeddings['labels']),
                'output_file': output_path,
            }
            
            metadata_path = os.path.join(args.output_dir, f"{base_name}_metadata.pt")
            torch.save(metadata, metadata_path)
            print(f"Metadata saved to: {metadata_path}")

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Processed domains: {args.domains}")
    print(f"Features saved in: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
