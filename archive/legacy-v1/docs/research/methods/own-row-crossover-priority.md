# Lai ghép dòng own theo cặp lớp bất đồng (`xo`) — sinh mẫu có chủ đích

**Trạng thái (21/10 ~12Z): FAIL** cổng gain — xo s42 ep15 private head 0,8321 vs xa 0,8246, Δ +0,0076 (benign +0,0063, tệ nhất +0,0113),
cần ≥ +0,010. Cơ chế đúng chỗ (cặp benign↔timeDelay ưu tiên ở 20/20 client) nhưng lợi trong nhiễu seed ⇒ dừng, không quét ρ/floor
([results §Vòng 24](../../methods/full-drift/results.md)). Owner 21/10: "ngoài Gaussian, đặt ưu tiên để sinh dữ liệu có chủ đích hơn".

## Vấn đề (bằng chứng)

[Probe 20](../../methods/full-drift/results.md) trên weights ep15 của headline (q3_zl la_kdpx s42): 41 % lỗi "teacher đúng, private sai" trên group là
benign↔timeDelay, phần lớn còn lại benign↔positionMirroring/dataReplay. Dòng own của client cho cùng cặp (cos gộp 0,94). Mẫu Gaussian gkd bất đồng
12,7 % nhưng ở cặp tốc độ (constantSpeedOffset→randomSpeedOffset; cos 0,09) ⇒ KL trên Gaussian không chạm chỗ private sai. Ưu tiên trong Gaussian
(top-KL, đối kháng, lớp/cặp) vẫn cos ≤ 0,12 — nhiễu liên tục ở feature tốc độ (thật là atom) làm mẫu nhìn như randomSpeedOffset.

## Cơ chế

Mỗi bước, từ round 2: 256 dòng, mỗi dòng = dòng own lớp a, mỗi feature thay bằng feature của một dòng own lớp b với xác suất 0,3; (a, b) theo
histogram (snapshot argmax, private argmax) trên 65.536 dòng own đầu epoch (90 %) + 10 % đều trên cặp client có. Dòng lai học KL(snapshot ‖ private)
như dòng Gaussian. Mẫu nằm giữa hai lớp private đang tách khác teacher, gần đa tạp thật (70 % feature từ một dòng thật) ⇒ private nhận ranh giới
toàn cục ở đúng vùng nhầm, vẫn giữ nhãn own trên dòng nguyên.

**R1:** 0 byte. **R2:** chỉ input-space + logit, private kiến trúc tùy ý. **Riêng tư:** không dòng nào rời xe; tri thức client khác chỉ qua snapshot.

## Kiểm

Arm `…_zl_xo` vs đối chứng xa (0,8246); đối chứng ưu tiên `…_zl_xou` (cặp đều). Cổng gain+worst +0,010; ưu tiên có tác dụng khi xo − xou ≥ +0,005.
Sai nếu xo < +0,010 (chỉ số phân bố của probe không dự báo utility, như cop) ⇒ dừng, không quét ρ/floor.

## Khác bản đã thử

- `fill` (A, FAIL): ưu tiên lớp hiếm local, tĩnh, trên Gaussian nhãn cứng. `rej` (B1): lọc mẫu Gaussian teacher bác bỏ — bỏ mẫu khó, không tạo mẫu mới.
- `cop` (FAIL): biên thật nhưng vẫn rút độc lập, bất đồng ở dataReplay↔benign, không timeDelay; không ưu tiên.
- Probe 17: tinh chỉnh Gaussian theo proxy (CE + BN), chỉ offline. Density ratio / MALA (archive): kéo về mật độ thật, không theo chỗ sai.
- Nguồn gần: hard-sample mining trong FedFTG (CVPR 2022) và Adaptive DeepInversion (CVPR 2020) sinh mẫu ở chỗ student lệch teacher bằng generator;
  ở đây mẫu lấy từ dữ liệu thật của chính xe nên không cần generator, không cần byte.
