# V10 — audited artifacts

14 arm đãaudit; 2 complete20round.
Source/config/weights/BN/payload/pred→clientCM/10metric đãaudit; liveW&B/status khôngthayproof. StageU, chưaR1/R2/DP.

| arm | seed | rounds | complete | cost ratio | final group score |
|---|---:|---:|---|---:|---:|
| e_ce_aug | 42 | 2 | False | 2.896522679868848 | unmeasured r20 |
| e_ce_aug | 43 | 2 | False | 2.9415876168207564 | unmeasured r20 |
| e_ce_matched | 42 | 2 | False | 2.9575663720838237 | unmeasured r20 |
| e_ce_matched | 43 | 2 | False | 3.0417264902632386 | unmeasured r20 |
| e_ce_raw | 42 | 20 | True | 1.0 | 0.685959797071513 |
| e_ce_raw | 43 | 20 | True | 1.0 | 0.6955807293717248 |
| e_fl | 42 | 2 | False | 2.9686678939403057 | unmeasured r20 |
| e_fl | 43 | 2 | False | 3.106630485419124 | unmeasured r20 |
| e_own | 42 | 2 | False | 2.9373428694892136 | unmeasured r20 |
| e_own | 43 | 2 | False | 3.0163371392251124 | unmeasured r20 |
| e_q | 42 | 2 | False | 2.32326038698142 | unmeasured r20 |
| e_q | 43 | 2 | False | 2.2483236757574434 | unmeasured r20 |
| e_shuffle | 42 | 2 | False | 2.9286440911285254 | unmeasured r20 |
| e_shuffle | 43 | 2 | False | 2.944050868970374 | unmeasured r20 |

## Locked gates

[Full-precision gates](../../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/probes/gates.json). Chưađủpairedcontrols thìpending; costFAILkhônglàqualityFAIL.

```json
{
  "energy_s42": {
    "cost": false,
    "quality": null,
    "reason": "cost stop before r20; quality unmeasured",
    "cost_ratio": 2.9686678939403057,
    "sampler_r1": {
      "two_noise_init": 2,
      "chains_per_init": 64,
      "transitions": 100,
      "movement": 0.05265625,
      "frozen_chains": 2,
      "mean_energy_difference_sd": 0.04205650507731728,
      "autocorrelation": {
        "1": 0.9426447153091431,
        "10": 0.563374400138855,
        "20": 0.2776646614074707
      },
      "energy_ess_per_chain": 4.197529077021925,
      "proposed_gate": false,
      "limited_finite_diagnostic": true
    }
  },
  "energy_s43": {
    "cost": false,
    "quality": null,
    "reason": "cost stop before r20; quality unmeasured",
    "cost_ratio": 3.106630485419124,
    "sampler_r1": {
      "two_noise_init": 2,
      "chains_per_init": 64,
      "transitions": 100,
      "movement": 0.063125,
      "frozen_chains": 1,
      "mean_energy_difference_sd": 0.1765235648569089,
      "autocorrelation": {
        "1": 0.9497731924057007,
        "10": 0.5955560207366943,
        "20": 0.28990429639816284
      },
      "energy_ess_per_chain": 4.035412129191543,
      "proposed_gate": false,
      "limited_finite_diagnostic": true
    }
  }
}
```

## Đủ10metric từng evaluation

[CSV](metrics-all.csv), [F1 theo lớp](per-class-f1.csv). Meanclientmetrics, khôngpooledconfusion.

| arm | seed | round | role | holdout | rule | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e_ce_raw | 42 | 5 | proxy | old | head | 0.789001 | 0.708968 | 0.789001 | 0.757766 | 0.629874 | 0.789001 | 0.789001 | 0.637879 | 0.789001 | 0.745100 |
| e_ce_raw | 42 | 5 | proxy | group | head | 0.784958 | 0.703704 | 0.784958 | 0.747923 | 0.621449 | 0.784958 | 0.784958 | 0.632034 | 0.784958 | 0.739874 |
| e_ce_raw | 42 | 10 | proxy | old | head | 0.777649 | 0.738435 | 0.777649 | 0.781540 | 0.683413 | 0.777649 | 0.777649 | 0.691318 | 0.777649 | 0.764901 |
| e_ce_raw | 42 | 10 | proxy | group | head | 0.788391 | 0.746669 | 0.788391 | 0.777204 | 0.682139 | 0.788391 | 0.788391 | 0.693816 | 0.788391 | 0.766625 |
| e_ce_raw | 42 | 15 | proxy | old | head | 0.776611 | 0.719503 | 0.776611 | 0.772346 | 0.698584 | 0.776611 | 0.776611 | 0.694121 | 0.776611 | 0.763996 |
| e_ce_raw | 42 | 15 | proxy | group | head | 0.785736 | 0.747396 | 0.785736 | 0.777824 | 0.700412 | 0.785736 | 0.785736 | 0.705068 | 0.785736 | 0.768816 |
| e_ce_raw | 42 | 20 | proxy | old | head | 0.788589 | 0.722349 | 0.788589 | 0.768542 | 0.672608 | 0.788589 | 0.788589 | 0.682481 | 0.788589 | 0.766257 |
| e_ce_raw | 42 | 20 | proxy | group | head | 0.791138 | 0.734486 | 0.791138 | 0.766297 | 0.674080 | 0.791138 | 0.791138 | 0.685960 | 0.791138 | 0.764346 |
| e_ce_raw | 43 | 5 | proxy | old | head | 0.789337 | 0.706826 | 0.789337 | 0.755548 | 0.631483 | 0.789337 | 0.789337 | 0.636220 | 0.789337 | 0.744377 |
| e_ce_raw | 43 | 5 | proxy | group | head | 0.786896 | 0.698597 | 0.786896 | 0.747658 | 0.625097 | 0.786896 | 0.786896 | 0.633252 | 0.786896 | 0.742535 |
| e_ce_raw | 43 | 10 | proxy | old | head | 0.786743 | 0.761678 | 0.786743 | 0.791537 | 0.677526 | 0.786743 | 0.786743 | 0.685488 | 0.786743 | 0.763437 |
| e_ce_raw | 43 | 10 | proxy | group | head | 0.787430 | 0.747395 | 0.787430 | 0.775867 | 0.669645 | 0.787430 | 0.787430 | 0.682524 | 0.787430 | 0.761482 |
| e_ce_raw | 43 | 15 | proxy | old | head | 0.781174 | 0.719610 | 0.781174 | 0.771827 | 0.702261 | 0.781174 | 0.781174 | 0.695965 | 0.781174 | 0.766041 |
| e_ce_raw | 43 | 15 | proxy | group | head | 0.787582 | 0.729702 | 0.787582 | 0.773635 | 0.699891 | 0.787582 | 0.787582 | 0.699504 | 0.787582 | 0.769118 |
| e_ce_raw | 43 | 20 | proxy | old | head | 0.799301 | 0.732432 | 0.799301 | 0.776635 | 0.684320 | 0.799301 | 0.799301 | 0.692499 | 0.799301 | 0.775187 |
| e_ce_raw | 43 | 20 | proxy | group | head | 0.801483 | 0.745697 | 0.801483 | 0.778556 | 0.681572 | 0.801483 | 0.801483 | 0.695581 | 0.801483 | 0.775210 |
