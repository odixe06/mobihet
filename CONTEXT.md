# CONTEXT — MobiHet v2 trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).
Lịch sử MobiHet v1 (Build1–P37, 20/50/100 client, 66 feature): [history/v1](history/v1/CONTEXT.md) — **giữ mọi lesson, đọc theo chủ đề**.

## Bài toán

IDS V2X 16 lớp; mỗi xe nhận là một client (5.343). MobiHet: mỗi xe giữ **model private không rời xe** (kiến trúc tùy ý) và chỉ trao
đổi một **proxy** nén (Q3 + LZMA2) ⇒ băng thông thấp, model khác nhau được. Repo này phát triển tiếp MobiHet trên dataset v4 với
**đúng protocol** của repo baseline [`odixe06/fl_dagsnet_veremi_v2`](https://github.com/odixe06/fl_dagsnet_veremi_v2)
([protocol](docs/research/protocol.md): 50 round, mọi xe, 1 epoch, DAGSNet W = 1, AdamW cosine, chấm test của chính xe) ⇒ số liệu so thẳng
với 15 baseline của repo đó (cùng code `core/`, cùng `scripts/cost.py`).

## Quyết định owner

- 30/10: tách repo `mobihet` mới (repo cũ → [`odixe06/mobihet-archive-veremi-v1`](https://github.com/odixe06/mobihet-archive-veremi-v1),
  archived, public; artifact DVC v1 trên OneDrive — [backup](docs/operations/backup.md)).
- 30/10 — hướng cải tiến: (1) train proxy song song như model thứ hai tốn tài nguyên/thời gian; (2) synthetic (Gaussian gpool) chưa hiệu quả;
  (3) cải thiện distill proxy ↔ private; (4) **cố gắng bỏ việc train proxy**. Kế hoạch: [improvement-plan](docs/research/improvement-plan.md).
- Kế thừa v2 (29–30/10): W = 1; gộp theo n_k; một seed 42; W&B online (project `mobihet-v4`); repo public, không đẩy data/PDF/credentials;
  commit không có `Co-Authored-By`.

## Trạng thái — 30/10/2026

- Arm hiện hành (v1 chốt) = `methods/mobihet.py`: `gkd + la + kdpx`, frz teacher, proxy q3_zl. Trên v4 (run v2 `fl2-mobihet`):
  r1 private 0,801 / proxy 0,441, **r1 1.335 s + eval 398 s, r2 > 36 phút** (FedAvg 224 s) ⇒ quá chậm cho 50 round; đang tìm nguyên nhân.
- Chi phí đo ([cost](docs/research/cost.md)): MobiHet 8,97 × forward/dòng (FedAvg 2,99), byte ~117 KB/xe/round (dense 1,6 MB).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Protocol chung (giống repo baseline) | [protocol](docs/research/protocol.md) |
| Cải tiến MobiHet: phân tích, thí nghiệm, cổng | [improvement-plan](docs/research/improvement-plan.md) |
| Chi phí tính toán / băng thông | [cost](docs/research/cost.md) |
| Dataset, DAGSNet, máy | [knowledge](knowledge/README.md) |
| Kaggle, lỗi kernel, backup | [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [backup](docs/operations/backup.md) |
| v1: thiết kế arm chốt, kết quả 20/50/100 client, synthetic P23–P37 (FAIL), mọi lesson | [history/v1 CONTEXT](history/v1/CONTEXT.md), [design](history/v1/docs/methods/full-drift/design.md), [LESSONS v1](history/v1/LESSONS.md) |
