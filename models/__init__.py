"""Neural network models for task segmentation."""

import torch

from .boundary_prediction import (
    BoundaryPredictionMLP,
    ModelConfig,
    collate_dict
)
from .boundary_prediction_transformer import (
    BoundaryPredictionTransformer,
    TransformerModelConfig
)
from .boundary_detr import (
    BoundaryDETR,
    BoundaryDETRConfig,
    create_boundary_detr
)
from .ms_tcn import (
    BoundaryMSTCN,
    build_boundary_mstcn
)
from .asformer import (
    BoundaryASFormer,
    build_boundary_asformer
)
from .detr_loss import (
    HungarianMatcher,
    SetCriterion,
    FocalSetCriterion,
    create_detr_criterion
)
from .encoders import TinyFASTEncoder, DinoFeatureEncoder
from .losses import (
    FocalLoss,
    SoftFocalLoss,
    FocalLossBCE,
    SoftBCELoss,
    WeightedBCELoss,
    MSEProbabilityLoss,
    get_loss_function
)


def create_model(config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Factory function to create models based on architecture type.

    Args:
        config: Model configuration (ModelConfig or TransformerModelConfig)
        device: Device to place model on

    Returns:
        Model instance on specified device

    Examples:
        >>> # MLP model (default)
        >>> config = ModelConfig()
        >>> model = create_model(config)

        >>> # Transformer model
        >>> config = TransformerModelConfig()
        >>> model = create_model(config)

        >>> # From dict with architecture flag
        >>> from omegaconf import OmegaConf
        >>> config_dict = OmegaConf.create({'architecture': 'cross_modal_transformer', ...})
        >>> model = create_model(config_dict)
    """
    # Check architecture type
    architecture = getattr(config, 'architecture', 'mlp')

    if architecture == 'cross_modal_transformer':
        # Create transformer config if needed
        if not isinstance(config, TransformerModelConfig):
            # Convert dict-like config to TransformerModelConfig, filtering relevant keys
            transformer_config_keys = {
                'architecture', 'vocab_size', 'token_embed_dim',
                'num_dino_frames', 'dino_feature_dim', 'num_visual_frames',
                'hidden_dims', 'dino_encoder_dropout', 'mlp_dropout', 'feature_noise_std',
                'visual_hidden_dim', 'visual_dropout',
                'cross_modal_dim', 'attention_dropout', 'num_attention_heads',
                'temporal_layers', 'temporal_dropout', 'temporal_feedforward_dim',
                'prediction_window_size', 'output_dropout', 'use_cls_token',
                'use_proprio', 'use_dino', 'fusion_mode',
                'transformer_dropout', 'embedding_dropout'
            }
            config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)
            config_dict = {k: v for k, v in config_dict.items() if k in transformer_config_keys}
            config = TransformerModelConfig(**config_dict)

        model = BoundaryPredictionTransformer(config=config, device=device)

    elif architecture == 'boundary_detr':
        # Create DETR model
        if not isinstance(config, BoundaryDETRConfig):
            # Convert dict-like config to BoundaryDETRConfig, filtering relevant keys
            detr_config_keys = {
                'architecture', 'vocab_size', 'token_embed_dim',
                'num_dino_frames', 'dino_feature_dim',
                'visual_hidden_dim', 'visual_dropout',
                'cross_modal_dim', 'attention_dropout', 'num_attention_heads',
                'd_model', 'num_encoder_layers', 'num_decoder_layers',
                'num_boundary_queries', 'detr_dropout', 'detr_feedforward_dim',
                'prediction_window_size', 'use_proprio', 'use_dino',
                'fusion_mode', 'transformer_dropout', 'embedding_dropout'
            }
            config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)
            config_dict = {k: v for k, v in config_dict.items() if k in detr_config_keys}
            config = BoundaryDETRConfig(**config_dict)

        model = create_boundary_detr(config=config, device=device)

    elif architecture == 'ms_tcn':
        # Create MS-TCN model
        # First get the full config to access ms_tcn parameters at top level
        full_config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)

        # Get MS-TCN hyperparameters from top-level config (not under model)
        ms_tcn_config = full_config_dict.get('ms_tcn', {})
        num_stages = ms_tcn_config.get('num_stages', 2)
        num_layers = ms_tcn_config.get('num_layers', 5)
        num_f_maps = ms_tcn_config.get('num_f_maps', 64)

        # Convert to ModelConfig (MS-TCN uses same config as MLP)
        if not isinstance(config, ModelConfig):
            model_config_keys = {
                'vocab_size', 'token_embed_dim', 'dino_feature_dim',
                'hidden_dims', 'prediction_window_size',
                'transformer_dropout', 'embedding_dropout',
                'dino_encoder_dropout', 'mlp_dropout', 'feature_noise_std',
                'use_proprio', 'use_dino', 'fusion_mode'
            }

            # If config has a 'model' key, use that for model-specific params
            if 'model' in full_config_dict and isinstance(full_config_dict['model'], dict):
                model_dict = full_config_dict['model']
                filtered_config = {k: v for k, v in model_dict.items() if k in model_config_keys}
            else:
                filtered_config = {k: v for k, v in full_config_dict.items() if k in model_config_keys}

            config = ModelConfig(**filtered_config)

        model = build_boundary_mstcn(
            config=config,
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            device=device
        )

    elif architecture == 'asformer':
        # Create ASFormer model
        # First get the full config to access asformer parameters at top level
        full_config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)

        # Get ASFormer hyperparameters from top-level config (not under model)
        asformer_config = full_config_dict.get('asformer', {})
        num_decoder_stages = asformer_config.get('num_decoder_stages', 3)
        num_encoder_layers = asformer_config.get('num_encoder_layers', 6)
        num_decoder_layers = asformer_config.get('num_decoder_layers', 3)
        num_f_maps = asformer_config.get('num_f_maps', 64)
        num_heads = asformer_config.get('num_heads', 4)
        attention_type = asformer_config.get('attention_type', 'normal')

        # Convert to ModelConfig (ASFormer uses same config as MLP)
        if not isinstance(config, ModelConfig):
            model_config_keys = {
                'vocab_size', 'token_embed_dim', 'dino_feature_dim',
                'hidden_dims', 'prediction_window_size',
                'transformer_dropout', 'embedding_dropout',
                'dino_encoder_dropout', 'mlp_dropout', 'feature_noise_std',
                'use_proprio', 'use_dino', 'fusion_mode'
            }

            # If config has a 'model' key, use that for model-specific params
            if 'model' in full_config_dict and isinstance(full_config_dict['model'], dict):
                model_dict = full_config_dict['model']
                filtered_config = {k: v for k, v in model_dict.items() if k in model_config_keys}
            else:
                filtered_config = {k: v for k, v in full_config_dict.items() if k in model_config_keys}

            config = ModelConfig(**filtered_config)

        model = build_boundary_asformer(
            config=config,
            num_decoder_stages=num_decoder_stages,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_f_maps=num_f_maps,
            num_heads=num_heads,
            attention_type=attention_type,
            device=device
        )

    else:
        # Default to MLP
        if not isinstance(config, ModelConfig):
            # Convert to ModelConfig, filtering transformer-specific keys
            model_config_keys = {
                'vocab_size', 'token_embed_dim', 'dino_feature_dim',
                'hidden_dims', 'prediction_window_size',
                'transformer_dropout', 'embedding_dropout',
                'dino_encoder_dropout', 'mlp_dropout', 'feature_noise_std',
                'use_proprio', 'use_dino', 'fusion_mode'
            }
            config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)
            filtered_config = {k: v for k, v in config_dict.items() if k in model_config_keys}
            config = ModelConfig(**filtered_config)

        model = BoundaryPredictionMLP(config=config, device=device)

    model = model.to(device)
    return model


__all__ = [
    'BoundaryPredictionMLP',
    'BoundaryPredictionTransformer',
    'BoundaryDETR',
    'BoundaryMSTCN',
    'BoundaryASFormer',
    'ModelConfig',
    'TransformerModelConfig',
    'BoundaryDETRConfig',
    'create_model',
    'create_boundary_detr',
    'build_boundary_mstcn',
    'build_boundary_asformer',
    'collate_dict',
    'TinyFASTEncoder',
    'DinoFeatureEncoder',
    'FocalLoss',
    'SoftFocalLoss',
    'FocalLossBCE',
    'SoftBCELoss',
    'WeightedBCELoss',
    'MSEProbabilityLoss',
    'get_loss_function',
    'HungarianMatcher',
    'SetCriterion',
    'FocalSetCriterion',
    'create_detr_criterion',
]
