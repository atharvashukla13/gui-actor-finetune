import torch.nn as nn

class ActionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, embeddings):
        return self.classifier(embeddings)

class GroundingVerifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, patch_embeddings):
        return self.fc(patch_embeddings).squeeze(-1)

class GUIActorModel(nn.Module):
    def __init__(self, backbone, action_head, verifier):
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.verifier = verifier
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, bbox=None, labels=None, **kwargs):
        backbone_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox
        }

        if "pixel_values" in kwargs:
            backbone_inputs["pixel_values"] = kwargs["pixel_values"]

        outputs = self.backbone(**backbone_inputs)
        embeddings = outputs.last_hidden_state  # [B, N, H]
        patch_embeddings = embeddings[:, 1:, :]  # [B, N-1, H]

        action_logits = self.action_head(patch_embeddings)  # [B, N-1, 2]
        grounding_scores = self.verifier(patch_embeddings)  # [B, N-1]

        loss = None
        if labels is not None:
            labels = labels[:, 1:]  # [B, N-1] to match action_logits
            loss = self.loss_fn(action_logits.reshape(-1, 2), labels.reshape(-1))

        return {
            "loss": loss,
            "action_logits": action_logits,
            "grounding_scores": grounding_scores,
        }
