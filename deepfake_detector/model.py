"""
Model Definition: EfficientNet-B3 (CNN) + Bidirectional LSTM + Attention
=========================================================================
Improved architecture for deepfake video classification.

Architecture:
  1. EfficientNet-B3 (pretrained on ImageNet) — spatial feature extraction
     - ~12M params (vs ~25M for ResNeXt50) — much less overfitting on small datasets
     - 82% ImageNet top-1 (vs 77.6%) — stronger pretrained features
  2. AdaptiveAvgPool2d — reduces spatial dims to 1x1
  3. Bidirectional LSTM (2 layers) — temporal dependencies across frames
  4. Temporal Attention — learns which frames matter most
  5. FC Head with LayerNorm + Dropout — robust classification

Key improvements over ResNeXt50 baseline:
  - EfficientNet-B3 backbone (better accuracy/param ratio)
  - freeze_backbone() / unfreeze_backbone() for 2-phase training
  - Temporal attention pooling instead of naive mean
  - Deeper classification head with LayerNorm
  - Bidirectional LSTM for richer temporal modeling
"""

import torch
from torch import nn
from torchvision import models
import dataset_config as cfg


class TemporalAttention(nn.Module):
    """
    Attention mechanism over LSTM temporal outputs.
    Learns to weight important frames more than unimportant ones.
    This is critical for deepfake detection: manipulated artifacts
    may only appear in certain frames.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, lstm_output, return_weights=False):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim)
            return_weights: If True, also return attention weights
        Returns:
            context: (batch_size, hidden_dim) — attention-weighted summary
            attn_weights: (batch_size, seq_len) — attention weights (if return_weights=True)
        """
        # Compute attention weights: (batch, seq_len, 1)
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum: (batch, hidden_dim)
        context = torch.sum(attn_weights * lstm_output, dim=1)

        if return_weights:
            return context, attn_weights.squeeze(-1)  # (batch, seq_len)
        return context


class Model(nn.Module):
    """
    EfficientNet-B3 + Bidirectional LSTM + Temporal Attention for deepfake detection.

    Args:
        num_classes (int): Number of output classes (default: 2 for REAL/FAKE).
        latent_dim (int): Feature dimension from CNN backbone (default: 1536 for EffNet-B3).
        lstm_layers (int): Number of LSTM layers (default: 2).
        hidden_dim (int): LSTM hidden state dimension (default: 256).
        bidirectional (bool): Whether LSTM is bidirectional (default: True).
        dropout (float): Dropout rate (default: 0.4).
    """

    def __init__(
        self,
        num_classes=cfg.NUM_CLASSES,
        latent_dim=cfg.LATENT_DIM,
        lstm_layers=cfg.LSTM_LAYERS,
        hidden_dim=cfg.HIDDEN_DIM,
        bidirectional=cfg.BIDIRECTIONAL,
        dropout=cfg.DROPOUT,
    ):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm_direction = 2 if bidirectional else 1

        # ─── CNN Backbone: EfficientNet-B3 (pretrained on ImageNet) ───
        # EfficientNet-B3: ~12M params, 82% ImageNet top-1
        # Output feature dim: 1536
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        # Split features into blocks for selective unfreezing
        # EfficientNet features: Sequential of 9 blocks (indices 0-8)
        #   0: Initial Conv (stem)
        #   1-4: Early MBConv blocks (general features)
        #   5-6: Mid MBConv blocks (unfrozen in Phase 2)
        #   7-8: Late MBConv blocks + head conv (unfrozen in Phase 2)
        features = list(backbone.features)

        # Early layers — stay frozen always (general low-level features)
        self.backbone_early = nn.Sequential(*features[:5])
        # Mid layers — unfrozen in Phase 2
        self.backbone_mid = nn.Sequential(*features[5:7])
        # Late layers — unfrozen in Phase 2
        self.backbone_late = nn.Sequential(*features[7:])

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Temporal module: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Temporal attention
        lstm_output_dim = hidden_dim * self.lstm_direction
        self.temporal_attention = TemporalAttention(lstm_output_dim)

        # Classification head: deeper with LayerNorm (works with any batch size)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.6),  # lighter dropout in last layer
            nn.Linear(128, num_classes),
        )

        # Initialize classifier weights properly
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier layers with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        """
        Freeze the entire CNN backbone.
        Used in Phase 1: only LSTM + attention + classifier train.
        """
        for param in self.backbone_early.parameters():
            param.requires_grad = False
        for param in self.backbone_mid.parameters():
            param.requires_grad = False
        for param in self.backbone_late.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [FREEZE] Backbone frozen. Trainable params: {trainable:,}")

    def unfreeze_backbone(self):
        """
        Unfreeze mid and late blocks of the backbone for fine-tuning.
        Early layers remain frozen (general features, no need to retrain).
        Used in Phase 2.
        """
        for param in self.backbone_mid.parameters():
            param.requires_grad = True
        for param in self.backbone_late.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [UNFREEZE] Mid + Late blocks unfrozen. Trainable params: {trainable:,}")

    def forward(self, x, return_frame_logits=False, return_attention=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            return_frame_logits: If True, return per-frame logits before attention pooling
            return_attention: If True, return attention weights from temporal attention

        Returns:
            fmap: Feature maps from last CNN layer (for visualization / CAM)
            logits: Classification logits of shape (batch_size, num_classes)
            frame_logits: Per-frame logits of shape (batch_size, seq_len, num_classes) (if return_frame_logits=True)
            attn_weights: Attention weights of shape (batch_size, seq_len) (if return_attention=True)
        """
        batch_size, seq_length, c, h, w = x.shape

        # Reshape: merge batch and sequence dims for CNN processing
        x = x.view(batch_size * seq_length, c, h, w)

        # Extract spatial features using EfficientNet-B3 (split backbone)
        x = self.backbone_early(x)
        x = self.backbone_mid(x)
        fmap = self.backbone_late(x)

        # Global average pooling → (batch*seq, 1536, 1, 1) → (batch*seq, 1536)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)

        # LSTM temporal encoding
        x_lstm, _ = self.lstm(x)

        # Temporal attention pooling (learns which frames matter)
        if return_attention:
            context, attn_weights = self.temporal_attention(x_lstm, return_weights=True)
        else:
            context = self.temporal_attention(x_lstm)

        # Classification
        logits = self.classifier(context)

        # Prepare return values
        outputs = [fmap, logits]

        if return_frame_logits:
            # Get per-frame logits by applying classifier to each LSTM output
            # x_lstm shape: (batch, seq_len, hidden_dim * 2)
            frame_logits = self.classifier(x_lstm)  # (batch, seq_len, num_classes)
            outputs.append(frame_logits)

        if return_attention:
            outputs.append(attn_weights)

        return tuple(outputs) if len(outputs) > 2 else (outputs[0], outputs[1]) if len(outputs) == 2 else outputs[0]
