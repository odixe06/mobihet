# F50 — phương pháp đề xuất ở 50 round, checkpoint mỗi round

Owner 03/11 (sau [X20](results-x20.md)): chạy phương pháp đề xuất trước; chưa cần trộn kiến trúc; phải có checkpoint mỗi round.
Recipe khóa ở `scripts/f50_common.py`: **hp đúng X20** (kiểm bằng so khớp với `runs/x-study/batch.json`), chỉ khác số round:
round 1..50 của lịch cosine50 ([protocol](protocol.md)), headline r50, không chọn round theo test.

## Ô

| arm | ô | seed | vai trò |
|---|---|---|---|
| F0 | bank (core + input-bank) | 42 | phương pháp đề xuất |
| F1 | bank | 43 | phương pháp đề xuất |
| F2 | core (floor + bootstrap nén) | 42 | ablation bank |
| F3 | core | 43 | ablation bank, cùng recipe, launch 05/11 trên account mới (`scripts/run_f50_s43.py`) |

Quota 03/11 chỉ đủ 3 owner có ≥ 15,5 h (session 1 cap 11 h + session resume 2,5 h, ×1,15).

## Checkpoint và resume

- Mỗi round: `cms_rNNN.npz`, `rounds.jsonl`, snapshot server `models/rNNN.pt` (Q, bank, mọi tensor < 10M phần tử) như X20,
  **và state resume đầy đủ** (`--save-every 1`): mọi model riêng của xe (fp32), BN, trạng thái server, generator của
  driver/phương pháp và RNG torch CPU/CUDA từng rank (dropout); rank 0 `state.pt`, rank 1 `state_rank1.pt`, ≈8,5 GB.
  Preflight đầu tiên (chưa lưu RNG toàn cục) lệch từ round đầu sau resume → đã thêm RNG toàn cục vào state.
- Chỉ giữ state của round mới nhất: output Kaggle ~20 GB, giữ mỗi round 8,5 GB là không thể. Lần lưu xóa state cũ của rank
  rồi ghi tmp → rename (một bản trên đĩa; cửa sổ vài giây không có state nếu máy chết đúng lúc ghi).
- Resume bỏ dòng *.jsonl của round sau round đã lưu (crash giữa ghi dòng và lưu state) → lịch sử không trùng round.
- Session chạm budget (cap 11 h, Kaggle 12 h) kết thúc COMPLETE trước r50 → pipeline tự push session `<slug>-c2` cùng owner,
  mount output cũ, tiếp từ round đã lưu, mang cả thư mục run; tối đa 3 session. Session ERROR không tự resume (phân loại tay).
- Preflight local (`runs/f-study/preflight.json`): core và bank, chạy thẳng tới r4 so với dừng r2 → resume r3 → khôi phục state r2
  (giả lập crash sau dòng r3) → resume r4: history, CM, snapshot, state cuối **trùng bit**; mỗi round một dòng. Giới hạn:
  sm86, world 1, 2 xe; không chứng minh 2 rank T4. Trên Kaggle, audit kiểm chuỗi teacher r−1 → r qua ranh giới session.

## Byte/xe/50 round

core 1.000.439 bootstrap + 50 × 1.088 = **1.054.839**; bank + 50 × 66.688 = **4.334.839**; FedAvg/APFed ≈159,5 MB.

## Đọc kết quả (khóa trước launch)

1. Headline r50 của bank (F0/F1): đủ 10 metric gộp + client mean/p10/min, đặt cạnh bảng baseline r50 seed 42 của repo
   benchmark (`../fl_dagsnet_veremi_v2/docs/research/results.md`; APFed client f1_weighted .9169, mean .8663, p10 .7533).
2. Giữ tầng bank nếu ở r50 bank − core vượt spread (low256 .003 hoặc p10 .006) ở **cả hai seed**; seed 42 đọc trước,
   kết luận sau core s43. Nếu không: phương pháp = core (1,05 MB/xe/50 round).
3. Đọc quỹ đạo r20/r30/r40/r50 của hiệu bank − core; r20 của F so với X cùng ô chỉ là kiểm tra nhất quán (khác session).
4. Không thêm thành phần, không đổi hp sau khi xem số.
