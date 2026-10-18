# CountSketch EF sửa overshoot

**Trạng thái (17/10):** FAIL (không nối).

## Cơ chế
Nén update proxy bằng CountSketch + error feedback, line-search chặn overshoot trên patch đã decode.

## Bằng chứng (audited)
Co norm PASS; small r20 hai seed quality FAIL; medium r17/r14 kém own. Decoded patch chỉ đổi 0,638%/7,818% tọa độ.

## Bài học
EF trong sketch-space không mặc định co; co norm không chứng minh teacher fidelity. q4 là đối chứng mạnh hơn.

## Nguồn
Nội bộ: [report](../../methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results/report.md).
