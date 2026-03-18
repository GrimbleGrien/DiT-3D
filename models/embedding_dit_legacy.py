import torch
import torch.nn as nn

from models.dit import DiTBlock, LabelEmbedder, TimestepEmbedder
from models.point_mae import MaskedEmbedder


class EmbeddingDiTLegacy(nn.Module):
    def __init__(
        self,
        embedding_dim=384,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        use_mae=False,
        mae_config_path=None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding_proj = nn.Linear(embedding_dim, hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)

        self.use_mae = use_mae
        if self.use_mae:
            class Args:
                pass

            args = Args()
            args.mae_config = mae_config_path or "configs/pretrainMAE.yaml"
            self.mae_embedder = MaskedEmbedder(args, hidden_size=hidden_size)
        else:
            self.mae_embedder = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_proj = nn.Linear(hidden_size, embedding_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.final_proj.weight, 0)
        if self.final_proj.bias is not None:
            nn.init.constant_(self.final_proj.bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, y, mae=None):
        x = x.squeeze(-1)
        x = self.embedding_proj(x).unsqueeze(1)
        x = x + self.pos_embed

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)

        if self.use_mae and mae is not None:
            mae_embed = self.mae_embedder(mae)
        else:
            mae_embed = torch.zeros_like(t_emb)

        c = t_emb + y_emb + mae_embed

        for block in self.blocks:
            x = block(x, c)

        x = x.squeeze(1)
        x = self.final_proj(x)
        return x.unsqueeze(-1)


def EmbeddingDiTLegacy_S_4(**kwargs):
    return EmbeddingDiTLegacy(**kwargs)
