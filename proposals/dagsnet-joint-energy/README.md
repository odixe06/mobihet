# Joint-energy bằng DAGSNet

[Danh mục](../README.md) · [Bài học](lessons.md)

Nhãn lịch sử: K4. Bản không base measure bị bác bỏ về normalizability trước train.
V10 Gaussian-base+MH đã triển khai, pull/audit cảhai seed; **costFAIL vàr1samplerFLFAIL**.

## Cơ chế FL

DAGSNet tạo năng lượng chung feature/label; sinh input synthetic tại xe, tham số model được gộp.

## Tài liệu liên quan

- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)
- [probe_v3_energy.py](../../scripts/probe_v3_energy.py)
- [energy-probe.json](../../papers/v3/energy-probe.json)

Mọi learned model giữ DAGSNet; không raw record rời xe. Chưa có bằng chứng cải thiện F1/R1/R2.

## Triển khai V10

CommonGaussian-base+MH/teacher-first/cost2×/samplerprobe đãchốt;
[manifest](../../docs/research/pilot-v10-energy.md), [source](../../mh/v10_energy.py),
[driver](../../mh/v10_run.py), [verifier](../../scripts/verify_teacher.py).
Local7arm/3exactresume/strictmodel/BN/payload/prediction PASS; khôngteacherF1/mixingproof.
BảnunbasedLebesgueR66 vẫnFAIL; adaptationmới hợp lệkhôngphủđịnhbằngchứng cũ.
[Report](../../papers/v10/report.md), [locked gates](../../papers/v10/gates.json).
12candidate dừngtheocost sau2round;2CErawreference đủ20round. Qualityr20energy chưađo;
khôngprivate/contact/rerun y nguyên.
