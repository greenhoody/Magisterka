import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    """
    Klasyczny blok ResNet: Conv3x3-BN-PReLU-Conv3x3-BN + skip.
    Bez zmiany rozmiaru map cech (stride=1), utrzymujemy HxW.
    """
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.prelu = nn.PReLU(num_parameters=ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        # aktywację po zsumowaniu zostawiamy „wprelu” w następnym bloku,
        # dzięki czemu zachowujemy klasyczny układ (post-act)
        return out


class ResNetTrunk(nn.Module):
    """
    „Stem” 1x1 → Res-bloki o stałej liczbie kanałów; brak poolingów, stałe HxW.
    """
    def __init__(self, in_ch: int, base_ch: int, num_blocks: int, dilations=None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.PReLU(num_parameters=base_ch),
        )
        if dilations is None:
            dilations = [1] * num_blocks
        assert len(dilations) == num_blocks, "dilations musi mieć długość = num_blocks"

        self.blocks = nn.ModuleList([BasicResBlock(base_ch, d) for d in dilations])

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return x  # [B, base_ch, H, W]


class CNNPolicyResNet(nn.Module):
    """
    ResNet-CNN kompatybilny z Twoim A2C:
    - wejścia: spatial [B,C,H,W], non-spatial [B, D]
    - wyjścia: value [B,1], policy logits [B, actions]
    - metody: forward / act / evaluate_actions / get_action_probs
    """
    def __init__(
        self,
        spatial_shape,          # (C, H, W)
        non_spatial_inputs: int,
        actions: int,
        base_channels: int = 64,
        residual_blocks: int = 8,
        dilations=None,         # np. [1,1,1,2,2,4,4,1] dla większego zasięgu
        hidden_nodes: int = 256 # rozmiar warstwy gęstej dla gałęzi non-spatial i critica
    ):
        super().__init__()
        C, H, W = spatial_shape

        # ---- Trunk przestrzenny (ResNet) ----
        self.trunk = ResNetTrunk(in_ch=C, base_ch=base_channels,
                                 num_blocks=residual_blocks, dilations=dilations)

        # ---- Gałąź nieprzestrzenna ----
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # ---- Łączenie strumieni ----
        stream_size = base_channels * H * W + hidden_nodes
        self.linear1       = nn.Linear(stream_size, stream_size)

        # ---- Głowy A2C: value & policy ----
        self.critic_linear = nn.Linear(stream_size, hidden_nodes)
        self.actor_linear  = nn.Linear(stream_size, stream_size)
        self.critic        = nn.Linear(hidden_nodes, 1)
        self.actor         = nn.Linear(stream_size, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        """
        He (Kaiming) init dla Conv/Linear; BN: gamma=1, beta=0.
        Spójne z dotychczasowym stylem w Twojej sieci.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spatial_input, non_spatial_input):
        # 1) Spatial → ResNet trunk
        x = self.trunk(spatial_input)        # [B, base_ch, H, W]
        x = x.flatten(start_dim=1)           # [B, base_ch*H*W]

        # 2) Non-spatial → dense
        y = self.linear0(non_spatial_input)  # [B, hidden_nodes]
        y = F.relu(y, inplace=True)

        # 3) Połączenie strumieni → wspólne przetwarzanie
        xy = torch.cat([x, y], dim=1)        # [B, stream_size]
        z  = self.linear1(xy)
        z  = F.relu(z, inplace=True)

        # 4) Głowy
        lc = self.critic_linear(z)
        la = self.actor_linear(z)
        value  = self.critic(lc)             # [B, 1]
        policy = self.actor(la)              # [B, actions]
        return value, policy

    # ====== API zgodne z Twoim CNNPolicy ======
    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, logits = self(spatial_input, non_spatial_input)
        if action_mask is not None:
            logits[~action_mask] = float("-inf")  # maskowanie akcji nielegalnych
        action_probs = F.softmax(logits, dim=1)
        return values, action_probs

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(
            spatial_inputs, non_spatial_input, action_mask=action_mask
        )
        actions = action_probs.multinomial(1)

        # Bezpieczeństwo: unikaj wyboru akcji o p=0 po masce
        for i, action in enumerate(actions):
            a = action
            while not action_mask[i][a]:
                a = action_probs[i].multinomial(1)
            actions[i] = a
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float("-inf")
        log_probs = F.log_softmax(policy, dim=1)
        probs     = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float("-inf"),
                                torch.tensor(0.0, device=log_probs.device),
                                log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy
