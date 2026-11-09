# L50 — Core lossless CDF, full50round, seed42

Owner04/11/2026 yêu cầu chạy đủ50round, 80train/20test như Core chính. Không dùng bootstrap64 hoặc Q8.

## Recipe khóa

- Core F2/X0: DAGSNetW1,5343xe, floor16synthetic, copula256bin, Qmeanfp16, noRealKD/noInputBank.
- Split v4 hiện hành: 37.361.273trainrow (foldtrain+val),9.347.679testrow (20% cuối timeline từngxe).
  Không chia ngẫu nhiên lại hoặc dùng officialtest để chọn HP/round; headline r50.
- Seed42;1epoch local/round,batch64,AdamW reset,AMPfp16/clip1,cosine50 giữ đúng [protocol](../../../docs/research/protocol.md).
- Chạy mới từr1, không gọi checkpoint C10 thành resume của phương pháp50round.

## Thay đổi duy nhất

[methods/mobihet_lossless.py](../../../methods/mobihet_lossless.py) kế thừa Core WLive.
Sau bootstrapcompact hiện hành, encode CDFuint16 dạng littleendian, zlib6, header32B (`MCDF`,version,shape,rawlen,bodylen,CRC32).
Client decode packet thật rồi dựng lại CDFfloat32/massfloat32 trên thiết bị train.
Assert CDF/mass trùng bit với Core **ngay tại bootstrap trên mỗi rank**; không đổi mean/Cholesky/prior hoặc RNG.
Train/synthetic dùng tensor đãdecode. Lưu `cdf_rank{0,1}.bin` và metadatahash/byte/bitexact cho audit.

CDF630784B → zlibbody267379B +header32B theo C10; bootstrap dự kiến up271031/down366035=637066B.
50roundQ=54400B → **691466B/xe**, so Core1054839B: giảm363373B (~34,45%). Đây là dự kiến trước L50,
phải lấy packet thật khi audit. SecAgg accounting dùng uplink dày1655872B → tổng2076307B/xe;
chưa thực thi SecAgg/DP/transport/ACK/retransmission. Lossless chỉ downlink, không cản cộng fixedvector uplink.

## Gate trước launch

- [test codec](../../../tests/test_lossless_cdf.py): roundtrip integerexact,CRC và truncationcorruption phải FAIL.
- [preflight](../../../scripts/preflight_l50.py): realfixture2xe/173trainrow,W1/compiled,4round;
  straight vsresume r2/crashrollbackr2 trùng history/CM/snapshot/fullstate; so originalCore trùng methodstate/snapshot/history ngoài accounting.
- SourceSHA chụp trước/sau,không sửa source saufreeze. Localworld1/sm86 không chứng minh T4world2.
- Quota/identity/slots/inputlive theoowner; Ownerchốt2account vì quota không đủ1account: session1r1–25,session2r26–50; mỗiaccountcap6,5h×1,15=7,475GPUh. Tổng dự trù14,95GPUh.
  Không launch khi quota không đủ; không giảmround hoặc bỏresume để lách gate.

## Vận hành

[scripts/run_l50.py](../../../scripts/run_l50.py) dùng launcher/resume/collector primitives F50, receipt riêng `runs/l50-study/`.
Notebookprivate,2T4imagepinned,npycachepublicSHAverified,W&Bonline,credentialprocessscoped.
Fullstate mỗiround; lateststate mỗirank khoảng8,5GB tổng, giữmộtbản như F50. Snapshotnhỏ mỗiround không phải fullprivatecheckpoint.
Session1 dừng cóchủđích ởr25 → pullfullstate2rank và artifact → archiveSHA → datasetprivate account2 → downloadverifySHA → importgateverified → requirestate25trướcdecodecache/trainr26. Session2 dừngr50. Budgetstop sớm/ERROR phảipull/đọc trước recovery; không khởiđộnglạir1.
Worker poll1200s qua codex-long-jobs; không modelpollwaiting.

## Audit/kết luận

[audit_l50.py](../../../scripts/audit_l50.py): decode packet thật cả2rank,CRC/hash/shape/CDF/mass,sourceoverlay,
CM→đủ10metric50round,teacherlag/signaturechain,protocol/IDs/support/finite/guards và logsessions.
So r50 với F2Coreseed42; báo pooled,clientmean/p10/low256,absent/rare,timeDelay,bytehai chiều gồmbootstrap.
Không chọn best round. Nếu cùngsource/runtime/RNG đãkhớp, lossless dự kiến không đổi tensor; sessionnoise vẫn cần đọc đúng phạmvi.
Không prediction→CM replay hoặc fullcontentfingerprintdataset; snapshot nhỏ không cho kiểm toàn bộ privateweights/BN.
