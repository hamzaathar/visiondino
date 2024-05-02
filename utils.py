import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class MultiCropWrapper(nn.Module):

    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):

        n_crops = len(x)
        # (n_samples * n_crops, 3, size, size)
        concatenated = torch.cat(x, dim=0)
        # (n_samples * n_crops, in_dim)
        cls_embedding = self.backbone(concatenated)
        logits = self.new_head(cls_embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

        return chunks


class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_crops, warmup_temp_teacher, temp_teacher,
                 warmup_temp_teacher_epochs, n_epochs, temp_student=0.1,
                 momentum_center=0.9):
        super().__init__()
        self.temp_student = temp_student
        self.momentum_center = momentum_center
        self.n_crops = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.temp_teacher_schedule = np.concatenate((
            np.linspace(warmup_temp_teacher,
                        temp_teacher, warmup_temp_teacher_epochs),
            np.ones(n_epochs - warmup_temp_teacher_epochs) * temp_teacher
        ))

    def forward(self, output_student, output_teacher, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        out_student = output_student / self.temp_student
        out_student = out_student.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.temp_teacher_schedule[epoch]
        out_teacher = F.softmax((output_teacher - self.center) / temp, dim=-1)
        out_teacher = out_teacher.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(out_teacher):
            for v in range(len(out_student)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(out_student[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(output_teacher)
        return total_loss


    @torch.no_grad()
    def update_center(self, output_teacher):
        """
        Update center used for teacher output.
        """
        batch_center = torch.cat(output_teacher).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

        # ema update
        #self.center = self.center * self.momentum_center + batch_center * (1 - self.momentum_center)


class Head(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=512,
        bottleneck_dim=256,
        n_layers=3,
        norm_last_layer=False,
    ):
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_parameters)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_parameters(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x
    
    
def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.

    Parameters
    ----------
    model : nn.Module
        Module.

    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)