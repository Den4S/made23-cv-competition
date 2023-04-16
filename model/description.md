```bash
model: efficientnet_v2_s
transfer learning
img_sz=320px, lr=1e-4, bs=32, num_epochs=7, AdamW optimizer, OneCycleLR scheduler, 6 augmentations.

clean labels:
efficientnet_v2_s, 320px model,
270 train images

ensemble:
[c1, c2] = softmax([f1_model1, f1_model2])
model1: efficientnet_v2_s, 320px, seed0
model2: efficientnet_v2_s, 360px, seed123
preds: c1 * softmax(model1_logits) + c2 * softmax(model2_logits)
```