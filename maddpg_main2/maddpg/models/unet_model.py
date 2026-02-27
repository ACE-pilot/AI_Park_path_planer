"""UNet model for 11x11 elevation map processing.
UNet encoder-decoder with skip connections processes elevation map,
MLP branch processes other observations, then features are fused.
"""
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

ELEVATION_OBS_SIZE = 11


class DoubleConv(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2D(in_ch, out_ch, 3, padding=1,
                               weight_attr=paddle.ParamAttr(
                                   initializer=nn.initializer.KaimingNormal()))
        self.conv2 = nn.Conv2D(out_ch, out_ch, 3, padding=1,
                               weight_attr=paddle.ParamAttr(
                                   initializer=nn.initializer.KaimingNormal()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNetEncoder(nn.Layer):
    """Small UNet for 11x11 elevation map -> feature vector."""

    def __init__(self, out_features=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(1, 16)       # [B,1,11,11] -> [B,16,11,11]
        self.pool1 = nn.MaxPool2D(2)         # -> [B,16,5,5]
        self.enc2 = DoubleConv(16, 32)       # -> [B,32,5,5]
        self.pool2 = nn.MaxPool2D(2)         # -> [B,32,2,2]

        # Bottleneck
        self.bottleneck = nn.Conv2D(32, 64, 2, padding=0,
                                    weight_attr=paddle.ParamAttr(
                                        initializer=nn.initializer.KaimingNormal()))
        # -> [B,64,1,1]

        # Decoder
        self.dec2_conv = DoubleConv(64 + 32, 32)   # up to 5x5 + skip enc2
        self.dec1_conv = DoubleConv(32 + 16, 16)   # up to 11x11 + skip enc1

        # Feature extraction: global average pool -> FC
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(16, out_features,
                            weight_attr=paddle.ParamAttr(
                                initializer=nn.initializer.XavierUniform()))

    def forward(self, elevation):
        """elevation: [B, 1, 11, 11]"""
        # Encoder
        e1 = self.enc1(elevation)                    # [B,16,11,11]
        e2 = self.enc2(self.pool1(e1))               # [B,32,5,5]
        bot = F.relu(self.bottleneck(self.pool2(e2))) # [B,64,1,1]

        # Decoder - upsample + skip
        up2 = F.interpolate(bot, size=[5, 5], mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(paddle.concat([up2, e2], axis=1))  # [B,32,5,5]

        up1 = F.interpolate(d2, size=[11, 11], mode='bilinear', align_corners=False)
        d1 = self.dec1_conv(paddle.concat([up1, e1], axis=1))  # [B,16,11,11]

        # Global average pool -> feature
        feat = self.gap(d1).squeeze(axis=[2, 3])     # [B,16]
        return self.fc(feat)                          # [B, out_features]


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        super().__init__()
        self.continuous_actions = continuous_actions

        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]
        if isinstance(act_dim, tuple):
            act_dim = act_dim[0]

        mlp_in = obs_dim - ELEVATION_OBS_SIZE ** 2

        # MLP branch for non-elevation features
        self.mlp_fc1 = nn.Linear(mlp_in, 64,
                                 weight_attr=paddle.ParamAttr(
                                     initializer=nn.initializer.XavierUniform()))
        self.mlp_fc2 = nn.Linear(64, 64,
                                 weight_attr=paddle.ParamAttr(
                                     initializer=nn.initializer.XavierUniform()))

        # UNet branch for elevation map
        self.unet = UNetEncoder(out_features=32)

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
        elev = obs[:, -ELEVATION_OBS_SIZE ** 2:].reshape(
            [-1, 1, ELEVATION_OBS_SIZE, ELEVATION_OBS_SIZE])

        mlp_feat = F.relu(self.mlp_fc2(F.relu(self.mlp_fc1(mlp_input))))
        unet_feat = self.unet(elev)

        combined = F.relu(self.shared_fc(paddle.concat([mlp_feat, unet_feat], axis=1)))
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


class UNetMAModel(parl.Model):
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
MAModel = UNetMAModel
