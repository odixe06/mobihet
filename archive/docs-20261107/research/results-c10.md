# C10 — nén Core, seed42, 10 round

04/11/2026: 4/4 kernel đủ10round; audit lại source/CM/teacher codec PASS. Cosine50,5343xe,DAGSNetW1,floor16.
Thiết kế/gate khóa trước: [screening-c10](screening-c10.md). Artifact: `runs/c10-study/{audit,analysis}.json`.

## Kết luận screening

C1 bootstrap64 PASS: −22,9% tổng byte r10, pooled/p10/low256 tăng nhỏ. C2 Q8 và C3combo FAIL low256/absent gate; không đưa Q8 vào Core.
Ưu tiên triển khai lossless CDF256 trước: Core dự kiến1011319→647914B/xe/r10 (−35,9%), không đổi tensor. Đây là measured codec + projected accounting, chưa chạy fullnetwork qua đường decode. C1 vẫn là ứng viên lossy, chưa thay mặc định50round.
Không sụp không có nghĩa giữ mọi cohort: Q8 làm absentaccuracy xe<256 giảm4,18điểm phầntrăm; combo giảm3,15điểm.

## Performance r10

| ô | pooled F1 | p10 | low256 mean F1 | timeDelay F1 | absent accuracy xe<256 | gate |
|---|---:|---:|---:|---:|---:|---|
| C0 core | 0.885973 | 0.652185 | 0.720229 | 0.280854 | 0.578135 | PASS |
| C1 boot64 | 0.887020 | 0.656474 | 0.722989 | 0.284175 | 0.577492 | PASS |
| C2 q8 | 0.885927 | 0.651689 | 0.712822 | 0.284978 | 0.536334 | FAIL |
| C3 combo | 0.886579 | 0.652312 | 0.717004 | 0.284148 | 0.546624 | FAIL |

## Đủ10metric pooled r10

| metric | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| accuracy | 0.910057 | 0.910032 | 0.909824 | 0.909848 |
| precision_macro | 0.896832 | 0.898789 | 0.896302 | 0.898592 |
| precision_micro | 0.910057 | 0.910032 | 0.909824 | 0.909848 |
| precision_weighted | 0.901091 | 0.900887 | 0.900919 | 0.900750 |
| recall_macro | 0.885696 | 0.884747 | 0.885714 | 0.884128 |
| recall_micro | 0.910057 | 0.910032 | 0.909824 | 0.909848 |
| recall_weighted | 0.910057 | 0.910032 | 0.909824 | 0.909848 |
| f1_macro | 0.885973 | 0.887020 | 0.885927 | 0.886579 |
| f1_micro | 0.910057 | 0.910032 | 0.909824 | 0.909848 |
| f1_weighted | 0.900712 | 0.901102 | 0.900698 | 0.900899 |

## Byte hai chiều/xe/r10, gồm bootstrap

| ô | thường | đường SecAgg dày | thêm lossless CDF (ước tính thường) |
|---|---:|---:|---:|
| C0 | 1,011,319 | 2,396,160 | 647,914 |
| C1 | 779,776 | 779,776 | 698,056 |
| C2 | 1,008,759 | 2,393,600 | 645,354 |
| C3 | 777,216 | 777,216 | 695,496 |

CDF toàn mạng256bin:630784→267379B (zlib6);64bin:157696→75976B. Hai rank cùng hash/byte, roundtripPASS.
Lossless đang là phép đo bổ sung trong kernel; tensor train không đi qua compressed CDF. Cột cuối là accounting triển khai dự kiến, chưa transport/header/ACK/crypto. Giữ mean/Cholesky fp16, prior và uplink hiện hành.
Bootstrap64 tính uplink dày bảo thủ; chưa đo sparse64. SecAgg là accounting theo vector dày của proposal; chưa fixed-point/overflow/crypto end-to-end.

## Delta và phân phối client

| ô−Core | Δpooled | Δp10 | Δlow256 | ΔtimeDelay | Δabsent | xe giảm | giảm>.05 | tăng>.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 | 0 | 0 |
| C1 | +0.001046 | +0.004289 | +0.002760 | +0.003320 | -0.000643 | 2573 | 488 | 503 |
| C2 | -0.000047 | -0.000496 | -0.007407 | +0.004124 | -0.041801 | 2679 | 472 | 426 |
| C3 | +0.000605 | +0.000127 | -0.003225 | +0.003294 | -0.031511 | 2631 | 512 | 473 |

Các count client là mô tả một seed; không dùng làm causalcount/significance. Rare=accuracy trên lớp có1–16trainrow, nhómxe<256; đầy đủ trong analysis.json.

## Rare và thời gian

| ô | rare accuracy xe<256 | train+eval10round (s) | max skip fraction |
|---|---:|---:|---:|
| C0 | 0.701002 | 6928.6 | 0.000000 |
| C1 | 0.705912 | 7150.4 | 0.000000 |
| C2 | 0.694456 | 7148.4 | 0.000000 |
| C3 | 0.695888 | 6802.9 | 0.000000 |

Thời gian toàn2T4, không phải latency xe; chưa gồm setup/compile/artifactI/O. Không suy speedup từ một session.

## F1 từng lớp r10

| lớp | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| benign | 0.852113 | 0.851759 | 0.851531 | 0.851533 |
| accelerationMultiplication | 0.863790 | 0.869572 | 0.863882 | 0.869359 |
| constantPositionOffset | 0.912137 | 0.912018 | 0.912567 | 0.912132 |
| constantSpeedOffset | 0.933509 | 0.933302 | 0.932540 | 0.933817 |
| dataReplay | 0.936998 | 0.937131 | 0.936540 | 0.935913 |
| dosAttack | 0.995063 | 0.995150 | 0.995123 | 0.995114 |
| feignedBraking | 0.927361 | 0.928204 | 0.926588 | 0.927938 |
| positionMirroring | 0.835170 | 0.838600 | 0.834674 | 0.837700 |
| randomPositionOffset | 0.985303 | 0.985344 | 0.985755 | 0.985245 |
| randomSpeedOffset | 0.971554 | 0.971688 | 0.971747 | 0.971429 |
| reversedHeading | 0.976836 | 0.977920 | 0.976487 | 0.977550 |
| suddenConstantSpeed | 0.781195 | 0.782782 | 0.779929 | 0.780090 |
| suddenStop | 0.981257 | 0.980469 | 0.980396 | 0.979750 |
| timeDelayAttack | 0.280854 | 0.284175 | 0.284978 | 0.284148 |
| trafficCongestionSybil | 0.994764 | 0.994546 | 0.994695 | 0.994620 |
| zeroSpeedReport | 0.947668 | 0.949655 | 0.947397 | 0.948922 |

## Giới hạn bằng chứng

Audit40CM→10metric/40snapshotnhỏ/40Q8packet (2ô×2rank×10round), source overlay/hash, teacherlag/chain và bootstrapmetadata. Snapshot nhỏ thiếu privateweights toàn mạng; chưaprediction→CM replay, chưa tái dựng streamuplink aggregate, chưa kiểm mọi tensor private/BN từ fullresume.
Skip được kiểm từ counters theo round; finite snapshot không chứng minh toàn bộ privatebank. Startup đã kiểm2T4/compiled cả2rank. Seed42/r10/officialtest đã đọc nhiều lần: screening, không xác nhận tương đương r50 hoặc mọi xe giữ chất lượng.
Không launch thêm, không kéo dài hoặc tinh chỉnh từ officialtest.
