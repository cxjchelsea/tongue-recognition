# D4-C.1-B Domain-Robust Stain Contract

- stain_contract_version: 1.1
- parent: stain_detection_v1 (1.0) preserved
- representation: black_masked_roi + letterbox 224 + ImageNet
- architecture: resnet18 ImageNet init（不从 v1 fine-tune）
- training: source supervised + source consistency + external unlabeled consistency
- forbidden: external pseudo labels / entropy minimization / test-based selection
- active policy switch requires TARGET_PASS + human confirmation
