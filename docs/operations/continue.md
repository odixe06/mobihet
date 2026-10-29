# Continue — bàn giao sống, 30/10/2026

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy

Không có job của repo này. Run `fl2-mobihet` (repo baseline, `trietbackup`, SHA cũ `6617261`) quá chậm và không resume đúng
⇒ owner được đề nghị dừng tay 30/10.

## Bước tiếp

1. Tìm nguyên nhân MobiHet chậm trên Kaggle (r2 > 36 phút vs FedAvg 224 s; local eager chỉ 2,6× FedAvg; GPU 100 %, bộ nhớ 96,8 %):
   nghi recompile / eager fallback (teacher forward thêm graph từ r2) hoặc nghẽn bộ nhớ. Chạy local `TORCH_LOGS=recompiles`.
2. Thí nghiệm cải tiến theo [improvement-plan](../research/improvement-plan.md) (local subset trước, cổng rồi mới Kaggle full).
3. Run full 50 round của arm thắng cổng; so với baseline repo `fl_dagsnet_veremi_v2` (cùng protocol).

## Việc tay (owner)

- (nếu chưa) dừng `trietbackup/fl2-mobihet` trên Kaggle.
