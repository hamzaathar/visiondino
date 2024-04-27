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
        batch_center = torch.sum(output_teacher, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(output_teacher) * dist.get_world_size())

        # ema update
        self.center = self.center * self.momentum_center + batch_center * (1 - self.momentum_center)
