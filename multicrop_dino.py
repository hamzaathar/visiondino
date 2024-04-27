class MultiCropWrapper(nn.Module):

    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):

        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, 3, size, size)
        cls_embedding = self.backbone(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.new_head(cls_embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

        return chunks
