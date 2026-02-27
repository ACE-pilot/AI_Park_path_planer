"""Self-Attention model for 11x11 elevation map processing.
Treats elevation map patches as tokens, applies multi-head self-attention,
fuses with MLP branch for non-elevation features.
"""
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

ELEVATION_OBS_SIZE = 11
NUM_PATCHES = ELEVATION_OBS_SIZE * ELEVATION_OBS_SIZE  # 121


class ElevationAttention(nn.Layer):
    """Self-Attention over 11x11 elevation patches."""

    def __init__(self, embed_dim=32, num_heads=4, out_features=32):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding: each cell -> embed_dim
        self.patch_proj = nn.Linear(1, embed_dim,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=nn.initializer.XavierUniform()))

        # Learnable positional encoding
        self.pos_embed = self.create_parameter(
            shape=[1, NUM_PATCHES, embed_dim],
            default_initializer=nn.initializer.Normal(std=0.02))

        # Multi-head self-attention
        self.attn = nn.MultiHeadAttention(embed_dim, num_heads,
                                          dropout=0.0)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2,
                      weight_attr=paddle.ParamAttr(
                          initializer=nn.initializer.XavierUniform())),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim,
                      weight_attr=paddle.ParamAttr(
                          initializer=nn.initializer.XavierUniform()))
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Output projection
        self.out_fc = nn.Linear(embed_dim, out_features,
                                weight_attr=paddle.ParamAttr(
                                    initializer=nn.initializer.XavierUniform()))

    def forward(self, elevation_flat):
        """elevation_flat: [B, 121]"""
        B = elevation_flat.shape[0]
        # [B, 121] -> [B, 121, 1] -> [B, 121, embed_dim]
        tokens = self.patch_proj(elevation_flat.unsqueeze(-1))
        tokens = tokens + self.pos_embed

        # Self-attention block
        attn_out = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Feed-forward block
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # Mean pool over all patches -> [B, embed_dim]
        feat = tokens.mean(axis=1)
        return self.out_fc(feat)  # [B, out_features]


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        super().__init__()
        self.continuous_actions = continuous_actions

        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]
        if isinstance(act_dim, tuple):
            act_dim = act_dim[0]

        mlp_in = obs_dim - ELEVATION_OBS_SIZE ** 2

        # MLP branch
        self.mlp_fc1 = nn.Linear(mlp_in, 64,
                                 weight_attr=paddle.ParamAttr(
                                     initializer=nn.initializer.XavierUniform()))
        self.mlp_fc2 = nn.Linear(64, 64,
                                 weight_attr=paddle.ParamAttr(
                                     initializer=nn.initializer.XavierUniform()))

        # Attention branch for elevation
        self.elev_attn = ElevationAttention(embed_dim=32, num_heads=4,
                                            out_features=32)

        # Fusion
        self.shared_fc = nn.Linear(64 + 32, 64,
                                   weight_attr=paddle.ParamAttr(
                                       initializer=nn.initializer.XavierUniform()))
        self.fc_out = nn.Linear(64, act_dim,
                                weight_attr=paddle.ParamAttr(
                                    initializer=nn.initializer.XavierUniform()))
        if continuous_actions:
            self.std_fc = nn.Linear(64, act_dim,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=nn.initializer.XavierUniform()))

    def forward(self, obs):
        mlp_input = obs[:, :-ELEVATION_OBS_SIZE ** 2]
        elev_flat = obs[:, -ELEVATION_OBS_SIZE ** 2:]

        mlp_feat = F.relu(self.mlp_fc2(F.relu(self.mlp_fc1(mlp_input))))
        attn_feat = self.elev_attn(elev_flat)

        combined = F.relu(self.shared_fc(paddle.concat([mlp_feat, attn_feat], axis=1)))
        means = self.fc_out(combined)

        if self.continuous_actions:
            act_std = self.std_fc(combined)
            act_std = paddle.clip(act_std, min=1e-6, max=1.0)
            return means, act_std
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super().__init__()
        self.fc1 = nn.Linear(critic_in_dim, 64,
                              weight_attr=paddle.ParamAttr(
                                  initializer=nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(64, 64,
                              weight_attr=paddle.ParamAttr(
                                  initializer=nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(64, 1,
                              weight_attr=paddle.ParamAttr(
                                  initializer=nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        inputs = paddle.concat(obs_n + act_n, axis=1)
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        Q = paddle.squeeze(Q, axis=1)
        return Q


class AttentionMAModel(parl.Model):
    def __init__(self, obs_dim, act_dim, obs_shape_n, act_shape_n,
                 continuous_actions=False):
        super().__init__()
        critic_in_dim = sum(s[0] for s in obs_shape_n) + sum(act_shape_n)

        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]
        if isinstance(act_dim, tuple):
            act_dim = act_dim[0]

        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


# Backward compatibility alias
MAModel = AttentionMAModel
