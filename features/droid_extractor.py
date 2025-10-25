"""Fast-DROID proprioceptive feature extraction and tokenization.

This module provides tokenization of robotic proprioceptive data using the
Fast-DROID method from Physical Intelligence.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor


# Configuration constants
FAST_MODEL = "physical-intelligence/fast"
DEFAULT_FRAME_FREQUENCY = 30
DEFAULT_MAX_TOKEN_LENGTH = 100
DEFAULT_VOCAB_SIZE = 2048


@dataclass
class FastTokenizerConfig:
    """
    Configuration for Fast-DROID tokenizer.

    Attributes:
        time_s: Time window in seconds for tokenization
        frame_frequency: Frame rate (Hz)
        action_dim: Dimension of action space (typically 14 for dual arm)
        vocab_size: Size of tokenizer vocabulary
    """
    time_s: float = 6.0
    frame_frequency: int = 30
    action_dim: int = 14
    vocab_size: int = DEFAULT_VOCAB_SIZE

    def tokenizer_id(self) -> str:
        """Generate unique identifier for this tokenizer configuration."""
        return (f"fast-tokenizer-time_{self.time_s:.2f}-"
                f"frame_frequency_{self.frame_frequency}-"
                f"action_dim_{self.action_dim}-"
                f"vocab_size_{self.vocab_size}")

    @property
    def window_size(self) -> int:
        """Get window size in frames."""
        return int(self.time_s * self.frame_frequency)


class DroidTokenEmbedder:
    """
    Fast-DROID trajectory tokenizer-based embedder.

    This class tokenizes proprioceptive action sequences into discrete tokens
    using a learned vocabulary. The tokenizer is trained on action differences
    (velocity) rather than absolute positions.

    Attributes:
        tokenizer: Fast-DROID tokenizer/processor
        proprio_stats: Normalization statistics (percentiles, mean, std, etc.)
        max_len: Maximum token sequence length
        frame_frequency: Frame rate for proprioception
        config: Tokenizer configuration
    """

    def __init__(
        self,
        config: Optional[FastTokenizerConfig] = None,
        tokenizer_path: Optional[str] = None,
        normalization_stats_path: Optional[str] = None,
        max_len: int = DEFAULT_MAX_TOKEN_LENGTH,
    ):
        """
        Initialize DROID token embedder.

        Args:
            config: Tokenizer configuration
            tokenizer_path: Path to saved tokenizer. If None, uses default
            normalization_stats_path: Path to normalization stats .npz file
            max_len: Maximum token length (pads or truncates)

        Example:
            >>> config = FastTokenizerConfig(time_s=6.0, vocab_size=2048)
            >>> embedder = DroidTokenEmbedder(config=config)
        """
        self.config = config or FastTokenizerConfig()
        self.max_len = max_len
        self.frame_frequency = self.config.frame_frequency

        # Load tokenizer
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = AutoProcessor.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
        else:
            # Use default from HuggingFace
            self.tokenizer = AutoProcessor.from_pretrained(
                FAST_MODEL,
                trust_remote_code=True
            )

        # Add special tokens
        if hasattr(self.tokenizer, 'bpe_tokenizer'):
            self.tokenizer.bpe_tokenizer.add_special_tokens({
                'pad_token': '[PAD]',
            })

        # Load normalization statistics
        self.proprio_stats = self._load_normalization_stats(normalization_stats_path)

    def _load_normalization_stats(
        self,
        stats_path: Optional[str]
    ) -> Dict[str, np.ndarray]:
        """
        Load proprioception normalization statistics.

        Args:
            stats_path: Path to .npz file with statistics

        Returns:
            Dictionary with normalization stats
        """
        if stats_path and Path(stats_path).exists():
            stats_data = np.load(stats_path)
            return dict(stats_data)
        else:
            # Return default stats (identity normalization)
            # These should be computed from training data in practice
            action_dim = self.config.action_dim
            return {
                'means': np.zeros(action_dim),
                'stds': np.ones(action_dim),
                'mins': -np.ones(action_dim),
                'maxs': np.ones(action_dim),
                'percentiles_1': -np.ones(action_dim),
                'percentiles_99': np.ones(action_dim),
            }

    def normalize_joint_actions(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Normalize joint action differences using percentile-based normalization.

        Maps actions to range [-1, 1] using 1st and 99th percentiles.

        Args:
            joint_positions: Action array of shape (N, action_dim)

        Returns:
            Normalized actions in range [-1, 1]
        """
        p1 = self.proprio_stats["percentiles_1"]
        p99 = self.proprio_stats["percentiles_99"]

        # Avoid division by zero
        denom = p99 - p1
        denom = np.where(denom == 0, 1.0, denom)

        return 2 * (joint_positions - p1) / denom - 1

    def embed_frames(
        self,
        joint_positions: np.ndarray,
        frames_of_interest: List[int],
        window_size: int,
    ) -> Dict[int, np.ndarray]:
        """
        Tokenize proprioceptive data for specified frame windows.

        For each frame of interest, extracts a window of actions starting
        at that frame and tokenizes them into discrete tokens.

        Args:
            joint_positions: Joint position trajectory of shape (N, action_dim)
            frames_of_interest: List of frame indices to tokenize
            window_size: Size of window for each tokenization

        Returns:
            Dictionary mapping frame index to token array

        Raises:
            AssertionError: If window size doesn't match config

        Example:
            >>> joint_pos = np.random.randn(1000, 14)
            >>> tokens = embedder.embed_frames(joint_pos, [0, 50, 100], window_size=180)
            >>> print(tokens[0].shape)  # (100,) - padded token sequence
        """
        # Validate window size
        if window_size % self.frame_frequency != 0:
            raise ValueError(
                f"Window size ({window_size}) must be a multiple of "
                f"frame frequency ({self.frame_frequency})"
            )

        expected_window = self.config.window_size
        if window_size != expected_window:
            raise ValueError(
                f"Window size ({window_size}) must equal time_s * frame_frequency "
                f"({expected_window}) that the tokenizer was trained on"
            )

        frames = sorted(set(frames_of_interest))
        embeddings_dict = {}

        # Get padding token ID
        if hasattr(self.tokenizer, 'bpe_tokenizer'):
            pad_token_id = self.tokenizer.bpe_tokenizer.pad_token_id
        else:
            pad_token_id = 0  # Default padding

        # Tokenize each window
        for frame_idx in frames:
            # Check if we have enough frames
            if frame_idx + window_size + 1 > len(joint_positions):
                # Skip or pad if insufficient data
                tokens = np.full(self.max_len, pad_token_id, dtype=np.int32)
                embeddings_dict[frame_idx] = tokens
                continue

            # Compute action differences (velocity)
            actions = np.diff(
                joint_positions[frame_idx:frame_idx + window_size + 1],
                axis=0
            )

            # Normalize actions
            actions = self.normalize_joint_actions(actions)

            # Tokenize
            try:
                token_output = self.tokenizer(actions)
                if isinstance(token_output, list):
                    tokens = np.array(token_output, dtype=np.int32).squeeze()
                else:
                    tokens = np.array(token_output, dtype=np.int32)

                # Pad or truncate to max_len
                if len(tokens) > self.max_len:
                    tokens = tokens[:self.max_len]
                else:
                    padding = np.full(
                        self.max_len - len(tokens),
                        pad_token_id,
                        dtype=np.int32
                    )
                    tokens = np.concatenate([tokens, padding])

                embeddings_dict[frame_idx] = tokens.astype(np.int32)

            except Exception as e:
                # If tokenization fails, use padding
                print(f"Warning: Tokenization failed for frame {frame_idx}: {e}")
                tokens = np.full(self.max_len, pad_token_id, dtype=np.int32)
                embeddings_dict[frame_idx] = tokens

        return embeddings_dict

    @staticmethod
    def compute_normalization_stats(
        trajectories: List[np.ndarray],
        samples_per_trajectory: int = 20,
        window_size: int = 180
    ) -> Dict[str, np.ndarray]:
        """
        Compute normalization statistics from training trajectories.

        This should be called on training data to compute the statistics
        used for normalization.

        Args:
            trajectories: List of joint position arrays (each shape (N, action_dim))
            samples_per_trajectory: Number of random windows per trajectory
            window_size: Size of each window

        Returns:
            Dictionary with normalization statistics

        Example:
            >>> # Compute stats from training data
            >>> stats = DroidTokenEmbedder.compute_normalization_stats(train_trajs)
            >>> # Save stats
            >>> np.savez('normalization_stats.npz', **stats)
        """
        action_dim = trajectories[0].shape[1]
        total_samples = len(trajectories) * samples_per_trajectory

        # Collect action differences
        action_data = np.zeros((total_samples, window_size, action_dim))

        for i, traj_positions in enumerate(trajectories):
            # Compute action differences
            actions = np.diff(traj_positions, axis=0)

            # Skip if trajectory too short
            if len(actions) < window_size:
                continue

            # Sample random windows
            max_start = len(actions) - window_size
            window_starts = np.random.randint(0, max_start, size=samples_per_trajectory)

            for j, start in enumerate(window_starts):
                action_data[i * samples_per_trajectory + j] = actions[start:start + window_size]

        # Compute statistics
        stats = {
            'means': action_data.mean(axis=(0, 1)),
            'stds': action_data.std(axis=(0, 1)),
            'mins': action_data.min(axis=(0, 1)),
            'maxs': action_data.max(axis=(0, 1)),
            'percentiles_1': np.percentile(action_data, 1, axis=(0, 1)),
            'percentiles_99': np.percentile(action_data, 99, axis=(0, 1)),
        }

        return stats

    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        if hasattr(self.tokenizer, 'bpe_tokenizer'):
            return len(self.tokenizer.bpe_tokenizer)
        elif hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        else:
            return self.config.vocab_size
