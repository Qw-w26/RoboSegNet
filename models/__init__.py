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
    'ModelConfig',
    'TransformerModelConfig',
    'BoundaryDETRConfig',
    'create_model',
    'create_boundary_detr',
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
