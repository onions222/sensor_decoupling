import torch
import torch.nn as nn
from typing import List, Optional


class _PatchNetFactory(nn.Module):
    """Helper to build small patch nets from a channel list.

    Usage: provide a list like [2,4,2] meaning conv -> bn -> relu layers
    with those intermediate channels, ending with a 1x1 conv to out_channels.
    """
    def __init__(self, in_channels: int = 1, channel_list: Optional[List[int]] = None, out_channels: int = 1):
        super().__init__()
        if channel_list is None:
            channel_list = [2, 4, 2]
        layers = []
        prev = in_channels
        for ch in channel_list:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        layers.append(nn.Conv2d(prev, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# TeacherModel kept as-is (large patch nets)
class _PatchNet_V13_Large_Teacher(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.odd_net = _PatchNet_V13_Large_Teacher()
        self.even_net = _PatchNet_V13_Large_Teacher()

    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)


class StudentModel(nn.Module):
    """StudentModel built from configurable channel list.

    Constructor expects `channel_list` for intermediate conv layers, e.g. [2,4,2].
    """
    def __init__(self, channel_list: Optional[List[int]] = None, in_channels: int = 1, out_channels: int = 1):
        super(StudentModel, self).__init__()
        if channel_list is None:
            channel_list = [2, 4, 2]
        self.odd_net = _PatchNetFactory(in_channels=in_channels, channel_list=channel_list, out_channels=out_channels)
        self.even_net = _PatchNetFactory(in_channels=in_channels, channel_list=channel_list, out_channels=out_channels)

    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)


def _scale_channels(base: List[int], mult: float) -> List[int]:
    scaled = [max(1, int(round(ch * mult))) for ch in base]
    return scaled


def make_student_model(channels: Optional[List[int]] = None,
                       multiplier: Optional[float] = None,
                       in_channels: int = 1,
                       out_channels: int = 1,
                       device: Optional[torch.device] = None) -> StudentModel:
    """Factory to create a StudentModel.

    - `channels`: explicit list of intermediate channels, e.g. [2,4,2].
    - `multiplier`: scale factor applied to default channels [2,4,2].
      If both provided, `channels` takes precedence.
    """
    base = [2, 4, 2]
    if channels is not None:
        ch_list = list(channels)
    elif multiplier is not None:
        ch_list = _scale_channels(base, multiplier)
    else:
        ch_list = base
    model = StudentModel(channel_list=ch_list, in_channels=in_channels, out_channels=out_channels)
    if device is not None:
        model.to(device)
    return model


if __name__ == "__main__":
    # Quick smoke test when running this file directly
    import torch
    print('Running quick local smoke test for models.py')
    s = make_student_model()
    t = TeacherModel()
    x = torch.randn(4, 1, 3, 5)
    is_odd = torch.tensor([0., 1., 0., 1.])
    print('Student out:', s(x, is_odd).shape)
    print('Teacher out:', t(x, is_odd).shape)
    s2 = make_student_model(multiplier=1.5)
    print('Student(mult=1.5) out:', s2(x, is_odd).shape)
