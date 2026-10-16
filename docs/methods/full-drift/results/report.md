# FullDrift — seed42 sàng lọc, group holdout (audited)

Primary: DML/FedAvg proxy head, Gaussian private gp. Test chỉ epoch 6, mô tả. Drift = ‖θ_k−θ_chung‖/‖θ_chung‖.

| arm | epoch | primary group | benign | private gp | private_ema gp | old primary | drift | test primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fd_dml | 1 | 0.1871 | 0.4934 | 0.5688 | 0.5701 | 0.1938 | 0.5923 |  |
| fd_dml | 2 | 0.6744 | 0.7344 | 0.5970 | 0.5833 | 0.6755 | 0.5553 |  |
| fd_dml | 3 | 0.7499 | 0.7411 | 0.6210 | 0.5960 | 0.7508 | 0.4973 |  |
| fd_dml | 4 | 0.7600 | 0.6958 | 0.6384 | 0.6077 | 0.7590 | 0.4514 |  |
| fd_dml | 5 | 0.7653 | 0.6210 | 0.6576 | 0.6165 | 0.7657 | 0.4147 |  |
| fd_dml | 6 | 0.7631 | 0.5646 | 0.6655 | 0.6257 | 0.7694 | 0.3840 | 0.7715 |
| fd_dml@e15 | 1 | 0.1871 | 0.4934 | 0.5688 | 0.5701 | 0.1938 | 0.5923 |  |
| fd_dml@e15 | 2 | 0.6744 | 0.7344 | 0.5970 | 0.5833 | 0.6755 | 0.5553 |  |
| fd_dml@e15 | 3 | 0.7499 | 0.7411 | 0.6210 | 0.5960 | 0.7508 | 0.4973 |  |
| fd_dml@e15 | 4 | 0.7600 | 0.6958 | 0.6384 | 0.6077 | 0.7590 | 0.4514 |  |
| fd_dml@e15 | 5 | 0.7653 | 0.6210 | 0.6576 | 0.6165 | 0.7657 | 0.4147 |  |
| fd_dml@e15 | 6 | 0.7631 | 0.5646 | 0.6655 | 0.6257 | 0.7694 | 0.3840 |  |
| fd_dml@e15 | 7 | 0.7645 | 0.5307 | 0.6704 | 0.6333 | 0.7723 | 0.3584 |  |
| fd_dml@e15 | 8 | 0.7641 | 0.4821 | 0.6799 | 0.6427 | 0.7728 | 0.3359 |  |
| fd_dml@e15 | 9 | 0.7687 | 0.4591 | 0.6891 | 0.6514 | 0.7770 | 0.3163 |  |
| fd_dml@e15 | 10 | 0.7665 | 0.4555 | 0.6872 | 0.6597 | 0.7767 | 0.2990 |  |
| fd_dml@e15 | 11 | 0.7668 | 0.4244 | 0.6968 | 0.6678 | 0.7770 | 0.2835 |  |
| fd_dml@e15 | 12 | 0.7672 | 0.4170 | 0.6889 | 0.6732 | 0.7783 | 0.2693 |  |
| fd_dml@e15 | 13 | 0.7658 | 0.3981 | 0.6989 | 0.6782 | 0.7770 | 0.2563 |  |
| fd_dml@e15 | 14 | 0.7629 | 0.3669 | 0.6997 | 0.6829 | 0.7764 | 0.2441 |  |
| fd_dml@e15 | 15 | 0.7652 | 0.3584 | 0.6960 | 0.6869 | 0.7795 | 0.2330 | 0.7769 |
| fd_dml_avgm | 1 | 0.1871 | 0.4933 | 0.5688 | 0.5701 | 0.1938 | 0.5923 |  |
| fd_dml_avgm | 2 | 0.5268 | 0.6709 | 0.5991 | 0.5839 | 0.5293 | 0.5551 |  |
| fd_dml_avgm | 3 | 0.4129 | 0.3942 | 0.6234 | 0.5967 | 0.4001 | 0.4891 |  |
| fd_dml_avgm | 4 | 0.0999 | 0.0000 | 0.6438 | 0.6100 | 0.0922 | 0.4026 |  |
| fd_dml_avgm | 5 | 0.0155 | 0.0001 | 0.6635 | 0.6208 | 0.0168 | 0.3371 |  |
| fd_dml_avgm | 6 | 0.0537 | 0.0000 | 0.6673 | 0.6307 | 0.0540 | 0.2946 | 0.0479 |
| fd_dml_hkd | 1 | 0.2015 | 0.4788 | 0.5617 | 0.5653 | 0.2127 | 0.5865 |  |
| fd_dml_hkd | 2 | 0.6673 | 0.7282 | 0.5928 | 0.5868 | 0.6718 | 0.5482 |  |
| fd_dml_hkd | 3 | 0.7471 | 0.7533 | 0.6156 | 0.6038 | 0.7469 | 0.4912 |  |
| fd_dml_hkd | 4 | 0.7615 | 0.7319 | 0.6397 | 0.6178 | 0.7660 | 0.4443 |  |
| fd_dml_hkd | 5 | 0.7683 | 0.6482 | 0.6606 | 0.6285 | 0.7741 | 0.4088 |  |
| fd_dml_hkd | 6 | 0.7661 | 0.6090 | 0.6710 | 0.6394 | 0.7725 | 0.3794 | 0.7745 |
| fd_dml_m4b | 1 | 0.3046 | 0.5411 | 0.6117 | 0.6132 | 0.3155 | 0.5826 |  |
| fd_dml_m4b | 2 | 0.7179 | 0.7478 | 0.6475 | 0.6291 | 0.7258 | 0.5438 |  |
| fd_dml_m4b | 3 | 0.7696 | 0.7689 | 0.6651 | 0.6410 | 0.7640 | 0.4840 |  |
| fd_dml_m4b | 4 | 0.7912 | 0.7546 | 0.6832 | 0.6526 | 0.7846 | 0.4345 |  |
| fd_dml_m4b | 5 | 0.8065 | 0.7610 | 0.6933 | 0.6616 | 0.7985 | 0.3964 |  |
| fd_dml_m4b | 6 | 0.8146 | 0.7593 | 0.7010 | 0.6711 | 0.8047 | 0.3665 | 0.8116 |
| fd_dml_m4b@e15 | 1 | 0.3046 | 0.5411 | 0.6117 | 0.6132 | 0.3155 | 0.5826 |  |
| fd_dml_m4b@e15 | 2 | 0.7179 | 0.7478 | 0.6475 | 0.6291 | 0.7258 | 0.5438 |  |
| fd_dml_m4b@e15 | 3 | 0.7696 | 0.7689 | 0.6651 | 0.6410 | 0.7640 | 0.4840 |  |
| fd_dml_m4b@e15 | 4 | 0.7912 | 0.7546 | 0.6832 | 0.6526 | 0.7846 | 0.4345 |  |
| fd_dml_m4b@e15 | 5 | 0.8065 | 0.7610 | 0.6933 | 0.6616 | 0.7985 | 0.3964 |  |
| fd_dml_m4b@e15 | 6 | 0.8146 | 0.7593 | 0.7010 | 0.6711 | 0.8047 | 0.3665 |  |
| fd_dml_m4b@e15 | 7 | 0.8217 | 0.7706 | 0.7034 | 0.6793 | 0.8115 | 0.3419 |  |
| fd_dml_m4b@e15 | 8 | 0.8290 | 0.7754 | 0.7120 | 0.6880 | 0.8157 | 0.3210 |  |
| fd_dml_m4b@e15 | 9 | 0.8368 | 0.7740 | 0.7167 | 0.6946 | 0.8200 | 0.3026 |  |
| fd_dml_m4b@e15 | 10 | 0.8364 | 0.7669 | 0.7161 | 0.7012 | 0.8197 | 0.2862 |  |
| fd_dml_m4b@e15 | 11 | 0.8405 | 0.7633 | 0.7234 | 0.7071 | 0.8263 | 0.2719 |  |
| fd_dml_m4b@e15 | 12 | 0.8441 | 0.7601 | 0.7234 | 0.7117 | 0.8283 | 0.2587 |  |
| fd_dml_m4b@e15 | 13 | 0.8440 | 0.7624 | 0.7297 | 0.7156 | 0.8284 | 0.2466 |  |
| fd_dml_m4b@e15 | 14 | 0.8486 | 0.7688 | 0.7279 | 0.7191 | 0.8325 | 0.2353 |  |
| fd_dml_m4b@e15 | 15 | 0.8510 | 0.7673 | 0.7323 | 0.7222 | 0.8369 | 0.2250 | 0.8322 |
| fd_dml_m4b_q4 | 1 | 0.3053 | 0.5469 | 0.6117 | 0.6132 | 0.3144 | 0.5826 |  |
| fd_dml_m4b_q4 | 2 | 0.7143 | 0.7475 | 0.6453 | 0.6283 | 0.7245 | 0.5435 |  |
| fd_dml_m4b_q4 | 3 | 0.7662 | 0.7714 | 0.6675 | 0.6407 | 0.7581 | 0.4837 |  |
| fd_dml_m4b_q4 | 4 | 0.7927 | 0.7540 | 0.6820 | 0.6518 | 0.7884 | 0.4344 |  |
| fd_dml_m4b_q4 | 5 | 0.8066 | 0.7722 | 0.6921 | 0.6615 | 0.7974 | 0.3965 |  |
| fd_dml_m4b_q4 | 6 | 0.8157 | 0.7689 | 0.7010 | 0.6707 | 0.8057 | 0.3663 | 0.8125 |
| fd_dml_m4b_sub4_q4 | 1 | 0.7654 | 0.7526 | 0.6463 | 0.6471 | 0.7614 | 0.2619 |  |
| fd_dml_m4b_sub4_q4 | 2 | 0.7992 | 0.7727 | 0.6883 | 0.6653 | 0.7931 | 0.2210 |  |
| fd_dml_m4b_sub4_q4 | 3 | 0.8186 | 0.7811 | 0.7058 | 0.6816 | 0.8094 | 0.2030 |  |
| fd_dml_m4b_sub4_q4 | 4 | 0.8317 | 0.7877 | 0.7176 | 0.6926 | 0.8184 | 0.1892 |  |
| fd_dml_m4b_sub4_q4 | 5 | 0.8429 | 0.7876 | 0.7259 | 0.7013 | 0.8303 | 0.1773 |  |
| fd_dml_m4b_sub4_q4 | 6 | 0.8485 | 0.7915 | 0.7319 | 0.7087 | 0.8330 | 0.1670 | 0.8277 |
| fd_dml_ntd | 1 | 0.1744 | 0.4344 | 0.5615 | 0.5605 | 0.1804 | 0.6076 |  |
| fd_dml_ntd | 2 | 0.6599 | 0.7284 | 0.5847 | 0.5790 | 0.6587 | 0.5672 |  |
| fd_dml_ntd | 3 | 0.7407 | 0.7497 | 0.5935 | 0.5880 | 0.7387 | 0.5112 |  |
| fd_dml_ntd | 4 | 0.7544 | 0.7214 | 0.5905 | 0.5911 | 0.7547 | 0.4658 |  |
| fd_dml_ntd | 5 | 0.7677 | 0.6979 | 0.6255 | 0.5968 | 0.7691 | 0.4304 |  |
| fd_dml_ntd | 6 | 0.7645 | 0.6570 | 0.6236 | 0.6035 | 0.7681 | 0.4005 | 0.7695 |
| fd_dml_prox | 1 | 0.2155 | 0.4809 | 0.5198 | 0.5242 | 0.2258 | 0.1178 |  |
| fd_dml_prox | 2 | 0.6021 | 0.7206 | 0.5553 | 0.5453 | 0.6202 | 0.1034 |  |
| fd_dml_prox | 3 | 0.6976 | 0.7383 | 0.5902 | 0.5692 | 0.7009 | 0.0897 |  |
| fd_dml_prox | 4 | 0.7305 | 0.7361 | 0.6178 | 0.5920 | 0.7317 | 0.0826 |  |
| fd_dml_prox | 5 | 0.7604 | 0.7268 | 0.6473 | 0.6108 | 0.7531 | 0.0787 |  |
| fd_dml_prox | 6 | 0.7624 | 0.6623 | 0.6527 | 0.6269 | 0.7561 | 0.0757 | 0.7510 |
| fd_dml_q4 | 1 | 0.1728 | 0.4514 | 0.5688 | 0.5701 | 0.1770 | 0.5923 |  |
| fd_dml_q4 | 2 | 0.6745 | 0.7346 | 0.5989 | 0.5838 | 0.6741 | 0.5547 |  |
| fd_dml_q4 | 3 | 0.7473 | 0.7499 | 0.6194 | 0.5959 | 0.7408 | 0.4977 |  |
| fd_dml_q4 | 4 | 0.7661 | 0.7544 | 0.6389 | 0.6067 | 0.7655 | 0.4515 |  |
| fd_dml_q4 | 5 | 0.7711 | 0.6212 | 0.6589 | 0.6165 | 0.7695 | 0.4147 |  |
| fd_dml_q4 | 6 | 0.7651 | 0.5968 | 0.6613 | 0.6243 | 0.7688 | 0.3837 | 0.7716 |
| fd_dml_sub16_q4 | 1 | 0.7811 | 0.7517 | 0.6477 | 0.6404 | 0.7729 | 0.1263 |  |
| fd_dml_sub16_q4 | 2 | 0.8009 | 0.7604 | 0.6801 | 0.6566 | 0.7977 | 0.1178 |  |
| fd_dml_sub16_q4 | 3 | 0.8107 | 0.7456 | 0.6911 | 0.6670 | 0.8069 | 0.1100 |  |
| fd_dml_sub16_q4 | 4 | 0.8187 | 0.7511 | 0.7031 | 0.6758 | 0.8148 | 0.1024 |  |
| fd_dml_sub16_q4 | 5 | 0.8252 | 0.7493 | 0.7145 | 0.6820 | 0.8251 | 0.0952 |  |
| fd_dml_sub16_q4 | 6 | 0.8270 | 0.7415 | 0.7139 | 0.6886 | 0.8290 | 0.0886 | 0.8118 |
| fd_dml_sub4 | 1 | 0.7514 | 0.7437 | 0.6090 | 0.6047 | 0.7458 | 0.2746 |  |
| fd_dml_sub4 | 2 | 0.7827 | 0.6936 | 0.6453 | 0.6220 | 0.7744 | 0.2353 |  |
| fd_dml_sub4 | 3 | 0.7883 | 0.6908 | 0.6642 | 0.6352 | 0.7828 | 0.2176 |  |
| fd_dml_sub4 | 4 | 0.7848 | 0.6595 | 0.6819 | 0.6466 | 0.7839 | 0.2030 |  |
| fd_dml_sub4 | 5 | 0.7947 | 0.6333 | 0.6961 | 0.6552 | 0.7950 | 0.1904 |  |
| fd_dml_sub4 | 6 | 0.7891 | 0.5946 | 0.7004 | 0.6643 | 0.7939 | 0.1790 | 0.7899 |
| fd_dml_sub4@e15 | 1 | 0.7514 | 0.7437 | 0.6090 | 0.6047 | 0.7458 | 0.2746 |  |
| fd_dml_sub4@e15 | 2 | 0.7827 | 0.6936 | 0.6453 | 0.6220 | 0.7744 | 0.2353 |  |
| fd_dml_sub4@e15 | 3 | 0.7883 | 0.6908 | 0.6642 | 0.6352 | 0.7828 | 0.2176 |  |
| fd_dml_sub4@e15 | 4 | 0.7848 | 0.6595 | 0.6819 | 0.6466 | 0.7839 | 0.2030 |  |
| fd_dml_sub4@e15 | 5 | 0.7947 | 0.6333 | 0.6961 | 0.6552 | 0.7950 | 0.1904 |  |
| fd_dml_sub4@e15 | 6 | 0.7891 | 0.5946 | 0.7004 | 0.6643 | 0.7939 | 0.1790 |  |
| fd_dml_sub4@e15 | 7 | 0.7877 | 0.5572 | 0.7038 | 0.6716 | 0.7965 | 0.1690 |  |
| fd_dml_sub4@e15 | 8 | 0.7850 | 0.4837 | 0.7074 | 0.6793 | 0.7967 | 0.1598 |  |
| fd_dml_sub4@e15 | 9 | 0.7918 | 0.4769 | 0.7135 | 0.6854 | 0.8041 | 0.1513 |  |
| fd_dml_sub4@e15 | 10 | 0.7845 | 0.4682 | 0.7135 | 0.6916 | 0.7979 | 0.1436 |  |
| fd_dml_sub4@e15 | 11 | 0.7896 | 0.4441 | 0.7193 | 0.6977 | 0.8037 | 0.1367 |  |
| fd_dml_sub4@e15 | 12 | 0.7904 | 0.4473 | 0.7173 | 0.7018 | 0.8075 | 0.1299 |  |
| fd_dml_sub4@e15 | 13 | 0.7913 | 0.4087 | 0.7217 | 0.7061 | 0.8085 | 0.1239 |  |
| fd_dml_sub4@e15 | 14 | 0.7923 | 0.4402 | 0.7195 | 0.7098 | 0.8087 | 0.1180 |  |
| fd_dml_sub4@e15 | 15 | 0.7926 | 0.4051 | 0.7211 | 0.7120 | 0.8079 | 0.1123 | 0.7980 |
| fd_dml_sub4_prox | 1 | 0.7248 | 0.7081 | 0.5984 | 0.5979 | 0.7241 | 0.0920 |  |
| fd_dml_sub4_prox | 2 | 0.7637 | 0.6110 | 0.6336 | 0.6166 | 0.7540 | 0.0720 |  |
| fd_dml_sub4_prox | 3 | 0.7701 | 0.6209 | 0.6455 | 0.6322 | 0.7622 | 0.0672 |  |
| fd_dml_sub4_prox | 4 | 0.7726 | 0.5727 | 0.6676 | 0.6459 | 0.7688 | 0.0640 |  |
| fd_dml_sub4_prox | 5 | 0.7831 | 0.5759 | 0.6850 | 0.6563 | 0.7830 | 0.0617 |  |
| fd_dml_sub4_prox | 6 | 0.7806 | 0.5689 | 0.6848 | 0.6662 | 0.7805 | 0.0599 | 0.7760 |
| fd_dml_sub4_q4 | 1 | 0.7492 | 0.7485 | 0.6098 | 0.6053 | 0.7458 | 0.2745 |  |
| fd_dml_sub4_q4 | 2 | 0.7771 | 0.7110 | 0.6464 | 0.6236 | 0.7701 | 0.2357 |  |
| fd_dml_sub4_q4 | 3 | 0.7828 | 0.6847 | 0.6632 | 0.6365 | 0.7808 | 0.2176 |  |
| fd_dml_sub4_q4 | 4 | 0.7837 | 0.6590 | 0.6815 | 0.6473 | 0.7862 | 0.2030 |  |
| fd_dml_sub4_q4 | 5 | 0.7919 | 0.6348 | 0.6966 | 0.6557 | 0.7974 | 0.1903 |  |
| fd_dml_sub4_q4 | 6 | 0.7863 | 0.5783 | 0.6969 | 0.6634 | 0.7965 | 0.1789 | 0.7931 |
| fd_dml_swappos | 1 | 0.2036 | 0.4844 | 0.6120 | 0.6046 | 0.2056 | 0.5871 |  |
| fd_dml_swappos | 2 | 0.7151 | 0.7501 | 0.6362 | 0.6189 | 0.7207 | 0.5505 |  |
| fd_dml_swappos | 3 | 0.7658 | 0.7722 | 0.6607 | 0.6303 | 0.7619 | 0.4918 |  |
| fd_dml_swappos | 4 | 0.7816 | 0.7817 | 0.6791 | 0.6411 | 0.7762 | 0.4429 |  |
| fd_dml_swappos | 5 | 0.8001 | 0.7892 | 0.6916 | 0.6526 | 0.7920 | 0.4047 |  |
| fd_dml_swappos | 6 | 0.8121 | 0.7894 | 0.6985 | 0.6641 | 0.8015 | 0.3739 | 0.8099 |
| fd_fedavg | 1 | 0.2465 | 0.4435 |  |  | 0.2748 | 0.6513 |  |
| fd_fedavg | 2 | 0.6813 | 0.7316 |  |  | 0.6836 | 0.5833 |  |
| fd_fedavg | 3 | 0.7490 | 0.6477 |  |  | 0.7540 | 0.5174 |  |
| fd_fedavg | 4 | 0.7561 | 0.5775 |  |  | 0.7648 | 0.4622 |  |
| fd_fedavg | 5 | 0.7702 | 0.5571 |  |  | 0.7769 | 0.4195 |  |
| fd_fedavg | 6 | 0.7724 | 0.5252 |  |  | 0.7784 | 0.3848 | 0.7655 |
| fd_fedavg@e15 | 1 | 0.2465 | 0.4435 |  |  | 0.2748 | 0.6513 |  |
| fd_fedavg@e15 | 2 | 0.6813 | 0.7316 |  |  | 0.6836 | 0.5833 |  |
| fd_fedavg@e15 | 3 | 0.7490 | 0.6477 |  |  | 0.7540 | 0.5174 |  |
| fd_fedavg@e15 | 4 | 0.7561 | 0.5775 |  |  | 0.7648 | 0.4622 |  |
| fd_fedavg@e15 | 5 | 0.7702 | 0.5571 |  |  | 0.7769 | 0.4195 |  |
| fd_fedavg@e15 | 6 | 0.7724 | 0.5252 |  |  | 0.7784 | 0.3848 |  |
| fd_fedavg@e15 | 7 | 0.7737 | 0.4589 |  |  | 0.7818 | 0.3570 |  |
| fd_fedavg@e15 | 8 | 0.7751 | 0.4395 |  |  | 0.7818 | 0.3330 |  |
| fd_fedavg@e15 | 9 | 0.7785 | 0.4069 |  |  | 0.7866 | 0.3123 |  |
| fd_fedavg@e15 | 10 | 0.7675 | 0.4091 |  |  | 0.7810 | 0.2939 |  |
| fd_fedavg@e15 | 11 | 0.7759 | 0.4119 |  |  | 0.7871 | 0.2780 |  |
| fd_fedavg@e15 | 12 | 0.7763 | 0.4092 |  |  | 0.7881 | 0.2631 |  |
| fd_fedavg@e15 | 13 | 0.7782 | 0.3904 |  |  | 0.7866 | 0.2498 |  |
| fd_fedavg@e15 | 14 | 0.7750 | 0.3776 |  |  | 0.7815 | 0.2370 |  |
| fd_fedavg@e15 | 15 | 0.7733 | 0.3356 |  |  | 0.7789 | 0.2255 | 0.7560 |
| fd_fedavg_sub16_q4 | 1 | 0.8064 | 0.6658 |  |  | 0.7955 | 0.1324 |  |
| fd_fedavg_sub16_q4 | 2 | 0.8306 | 0.6604 |  |  | 0.8223 | 0.1217 |  |
| fd_fedavg_sub16_q4 | 3 | 0.8374 | 0.6504 |  |  | 0.8334 | 0.1129 |  |
| fd_fedavg_sub16_q4 | 4 | 0.8414 | 0.6593 |  |  | 0.8327 | 0.1047 |  |
| fd_fedavg_sub16_q4 | 5 | 0.8570 | 0.6440 |  |  | 0.8555 | 0.0973 |  |
| fd_fedavg_sub16_q4 | 6 | 0.8527 | 0.6223 |  |  | 0.8476 | 0.0906 | 0.8148 |
| fd_fedavg_sub4 | 1 | 0.7615 | 0.6978 |  |  | 0.7529 | 0.2930 |  |
| fd_fedavg_sub4 | 2 | 0.7883 | 0.6515 |  |  | 0.7818 | 0.2441 |  |
| fd_fedavg_sub4 | 3 | 0.7917 | 0.5900 |  |  | 0.7931 | 0.2230 |  |
| fd_fedavg_sub4 | 4 | 0.8026 | 0.5375 |  |  | 0.8012 | 0.2067 |  |
| fd_fedavg_sub4 | 5 | 0.8197 | 0.5284 |  |  | 0.8191 | 0.1929 |  |
| fd_fedavg_sub4 | 6 | 0.8059 | 0.4639 |  |  | 0.8059 | 0.1809 | 0.7877 |
| fd_fedavg_sub4@e15 | 1 | 0.7615 | 0.6978 |  |  | 0.7529 | 0.2930 |  |
| fd_fedavg_sub4@e15 | 2 | 0.7883 | 0.6515 |  |  | 0.7818 | 0.2441 |  |
| fd_fedavg_sub4@e15 | 3 | 0.7917 | 0.5900 |  |  | 0.7931 | 0.2230 |  |
| fd_fedavg_sub4@e15 | 4 | 0.8026 | 0.5375 |  |  | 0.8012 | 0.2067 |  |
| fd_fedavg_sub4@e15 | 5 | 0.8197 | 0.5284 |  |  | 0.8191 | 0.1929 |  |
| fd_fedavg_sub4@e15 | 6 | 0.8059 | 0.4639 |  |  | 0.8059 | 0.1809 |  |
| fd_fedavg_sub4@e15 | 7 | 0.8108 | 0.5039 |  |  | 0.8105 | 0.1705 |  |
| fd_fedavg_sub4@e15 | 8 | 0.8128 | 0.4465 |  |  | 0.8138 | 0.1611 |  |
| fd_fedavg_sub4@e15 | 9 | 0.8222 | 0.4683 |  |  | 0.8225 | 0.1526 |  |
| fd_fedavg_sub4@e15 | 10 | 0.8073 | 0.4487 |  |  | 0.8120 | 0.1446 |  |
| fd_fedavg_sub4@e15 | 11 | 0.8193 | 0.4737 |  |  | 0.8228 | 0.1372 |  |
| fd_fedavg_sub4@e15 | 12 | 0.8154 | 0.4595 |  |  | 0.8185 | 0.1306 |  |
| fd_fedavg_sub4@e15 | 13 | 0.8184 | 0.4735 |  |  | 0.8223 | 0.1242 |  |
| fd_fedavg_sub4@e15 | 14 | 0.8169 | 0.4729 |  |  | 0.8228 | 0.1179 |  |
| fd_fedavg_sub4@e15 | 15 | 0.8142 | 0.4468 |  |  | 0.8217 | 0.1124 | 0.7891 |
| fd_fedavg_sub4_prox | 1 | 0.7314 | 0.6950 |  |  | 0.7307 | 0.0894 |  |
| fd_fedavg_sub4_prox | 2 | 0.7593 | 0.5517 |  |  | 0.7500 | 0.0701 |  |
| fd_fedavg_sub4_prox | 3 | 0.7616 | 0.5166 |  |  | 0.7504 | 0.0658 |  |
| fd_fedavg_sub4_prox | 4 | 0.7680 | 0.5281 |  |  | 0.7586 | 0.0635 |  |
| fd_fedavg_sub4_prox | 5 | 0.7814 | 0.5053 |  |  | 0.7725 | 0.0616 |  |
| fd_fedavg_sub4_prox | 6 | 0.7792 | 0.4894 |  |  | 0.7737 | 0.0600 | 0.7699 |
| fd_fedavg_sub4_q4 | 1 | 0.7563 | 0.6473 |  |  | 0.7520 | 0.2929 |  |
| fd_fedavg_sub4_q4 | 2 | 0.7782 | 0.6086 |  |  | 0.7747 | 0.2442 |  |
| fd_fedavg_sub4_q4 | 3 | 0.7845 | 0.5003 |  |  | 0.7854 | 0.2231 |  |
| fd_fedavg_sub4_q4 | 4 | 0.7826 | 0.4696 |  |  | 0.7864 | 0.2065 |  |
| fd_fedavg_sub4_q4 | 5 | 0.8140 | 0.4940 |  |  | 0.8144 | 0.1928 |  |
| fd_fedavg_sub4_q4 | 6 | 0.7991 | 0.4496 |  |  | 0.8028 | 0.1804 | 0.7800 |
| fd_gaussian | 1 | 0.6173 | 0.3651 | 0.6173 | 0.6402 | 0.6163 |  |  |
| fd_gaussian | 2 | 0.6032 | 0.3305 | 0.6032 | 0.6444 | 0.6042 |  |  |
| fd_gaussian | 3 | 0.5994 | 0.3228 | 0.5994 | 0.6402 | 0.6031 |  |  |
| fd_gaussian | 4 | 0.5926 | 0.3116 | 0.5926 | 0.6328 | 0.5954 |  |  |
| fd_gaussian | 5 | 0.5912 | 0.3096 | 0.5912 | 0.6225 | 0.5947 |  |  |
| fd_gaussian | 6 | 0.5882 | 0.2970 | 0.5882 | 0.6126 | 0.5906 |  | 0.6074 |
| fd_gaussian_m4b | 1 | 0.6462 | 0.4534 | 0.6462 | 0.6518 | 0.6441 |  |  |
| fd_gaussian_m4b | 2 | 0.6510 | 0.4433 | 0.6510 | 0.6580 | 0.6501 |  |  |
| fd_gaussian_m4b | 3 | 0.6585 | 0.4539 | 0.6585 | 0.6575 | 0.6585 |  |  |
| fd_gaussian_m4b | 4 | 0.6495 | 0.4313 | 0.6495 | 0.6508 | 0.6484 |  |  |
| fd_gaussian_m4b | 5 | 0.6479 | 0.4371 | 0.6479 | 0.6489 | 0.6493 |  |  |
| fd_gaussian_m4b | 6 | 0.6553 | 0.4395 | 0.6553 | 0.6452 | 0.6544 |  | 0.6671 |
| fd_gaussian_swappos | 1 | 0.6353 | 0.4374 | 0.6353 | 0.6678 | 0.6355 |  |  |
| fd_gaussian_swappos | 2 | 0.6363 | 0.4346 | 0.6363 | 0.6773 | 0.6347 |  |  |
| fd_gaussian_swappos | 3 | 0.6430 | 0.4358 | 0.6430 | 0.6811 | 0.6437 |  |  |
| fd_gaussian_swappos | 4 | 0.6381 | 0.4303 | 0.6381 | 0.6804 | 0.6383 |  |  |
| fd_gaussian_swappos | 5 | 0.6380 | 0.4355 | 0.6380 | 0.6803 | 0.6375 |  |  |
| fd_gaussian_swappos | 6 | 0.6404 | 0.4265 | 0.6404 | 0.6768 | 0.6382 |  | 0.6540 |

| so sánh (group; epoch theo cổng) | Δ primary | Δ benign | cổng (rule trong gates.json) |
|---|---:|---:|---|
| fd_dml_sub4 | +0.0259 | +0.0300 | True |
| fd_dml_prox | -0.0008 | +0.0977 | False |
| fd_dml_avgm | -0.7094 | -0.5646 | False |
| fd_fedavg_sub4 | +0.0335 | -0.0613 | False |
| fd_gaussian_swappos | +0.0522 | +0.1295 | True |
| fd_dml_sub4_q4 | -0.0028 | -0.0164 | False |
| fd_fedavg_sub4_q4 | -0.0068 | -0.0143 | False |
| fd_dml_sub16_q4 | +0.0407 | +0.1633 | True |
| fd_fedavg_sub16_q4 | +0.0536 | +0.1727 | True |
| fd_dml_sub4_prox | -0.0084 | -0.0258 | False |
| fd_fedavg_sub4_prox | -0.0267 | +0.0255 | False |
| fd_dml_sub4@e15 | +0.0274 | +0.0467 | True |
| fd_fedavg_sub4@e15 | +0.0410 | +0.1112 | True |
| fd_dml_q4 | +0.0019 | +0.0322 | True |
| fd_dml_m4b_q4 | +0.0011 | +0.0096 | True |
| fd_dml_m4b@e15 | +0.0858 | +0.4090 | True |
| fd_dml_m4b@s43 |  |  | None |
| fd_dml_m4b_sub4_q4@s43 |  |  | None |
| fd_dml_m4b_sub4_q4@s43_vs_fd_dml_sub4_q4@s43 |  |  | None |
| fd_dml_sub4_q4@s43 |  |  | None |
| fd_dml_swappos@s43 |  |  | None |
| fd_dml@e15:replay_ep1-6 | +0.0000 |  | mô tả |
| fd_dml_sub4@e15:replay_ep1-6 | +0.0000 |  | mô tả |
| fd_fedavg@e15:replay_ep1-6 | +0.0000 |  | mô tả |
| fd_fedavg_sub4@e15:replay_ep1-6 | +0.0000 |  | mô tả |
| fd_dml_m4b@e15:replay_ep1-6 | +0.0000 |  | mô tả |
| fd_dml:private_ema | -0.0398 | -0.0143 | False |
| fd_dml_sub4:private_ema | -0.0362 | -0.0056 | False |
| fd_dml_prox:private_ema | -0.0258 | -0.0028 | False |
| fd_gaussian:private_ema | +0.0244 | +0.0233 | True |
| fd_gaussian_swappos:private_ema | +0.0364 | +0.0160 | True |
| fd_dml_avgm:private_ema | -0.0365 | -0.0088 | False |
| fd_dml@e15:private_ema | -0.0398 | -0.0143 | False |
| fd_dml_sub4@e15:private_ema | -0.0362 | -0.0056 | False |
| fd_dml_sub16_q4:private_ema | -0.0252 | +0.0039 | False |
| fd_dml_sub4_prox:private_ema | -0.0187 | +0.0053 | False |
| fd_dml_sub4_q4:private_ema | -0.0335 | -0.0020 | False |
| fd_dml_m4b:private_ema | -0.0299 | -0.0300 | False |
| fd_dml_swappos:private_ema | -0.0344 | -0.0261 | False |
| fd_dml_ntd:private_ema | -0.0201 | +0.0048 | False |
| fd_dml_q4:private_ema | -0.0371 | -0.0124 | False |
| fd_dml_hkd:private_ema | -0.0316 | +0.0019 | False |
| fd_gaussian_m4b:private_ema | -0.0101 | -0.0085 | False |
| fd_dml_m4b_q4:private_ema | -0.0303 | -0.0293 | False |
| fd_dml_m4b@e15:private_ema | -0.0299 | -0.0300 | False |
| fd_dml_m4b_sub4_q4:private_ema | -0.0231 | -0.0171 | False |
| fd_dml_sub4_q4:private_gp | -0.0036 | -0.0037 | True |
| fd_dml_sub16_q4:private_gp | +0.0170 | +0.0266 | True |
| fd_dml_sub4_prox:private_gp | -0.0156 | +0.0021 | False |
| fd_dml_sub4@e15:private_gp | +0.0251 | +0.0272 | True |
| fd_dml_m4b:private_gp | +0.0355 | +0.1134 | True |
| fd_dml_swappos:private_gp | +0.0330 | +0.1070 | True |
| fd_dml_ntd:private_gp | -0.0418 | -0.0338 | False |
| fd_dml_q4:private_gp | -0.0041 | +0.0000 | True |
| fd_dml_hkd:private_gp | +0.0056 | +0.0005 | False |
| fd_gaussian_m4b:private_gp | +0.0671 | +0.1424 | True |
| fd_dml_m4b@e15:private_gp | +0.0363 | +0.1438 | True |
| fd_dml_m4b_q4:private_gp | +0.0000 | -0.0004 | True |
| fd_dml_m4b_sub4_q4:private_gp | +0.0309 | +0.0206 | True |
| fd_dml_m4b@s43:private_gp |  |  | None |
| fd_dml_m4b_sub4_q4@s43:private_gp |  |  | None |
| fd_dml_m4b_sub4_q4@s43:private_gp_vs_fd_dml_sub4_q4@s43 |  |  | None |
| fd_dml_sub4_q4@s43:private_gp |  |  | None |
| fd_dml_swappos@s43:private_gp |  |  | None |
| fd_dml_sub4:private_gp | +0.0350 | +0.0308 | mô tả |
| fd_dml_prox:private_gp | -0.0127 | +0.0319 | mô tả |
| fd_dml_avgm:private_gp | +0.0018 | -0.0006 | mô tả |
| fd_fedavg_vs_fd_dml | +0.0093 |  | mô tả |

## Góc nhìn client (epoch cuối, group): mean20client (client tệ nhất)

local = confusion đánh trọng số lại theo prior nhãn train của từng client (chỉ lệch nhãn).

| arm | epoch | model/rule | global F1 (min) | local F1 (min) | local acc (min) |
|---|---:|---|---|---|---|
| fd_dml | 6 | private/head | global_f1 0.5446 (min 0.3206) | local_f1 0.6281 (min 0.4248) | local_acc 0.8262 (min 0.6495) |
| fd_dml | 6 | private/gp | global_f1 0.6655 (min 0.5945) | local_f1 0.6371 (min 0.4914) | local_acc 0.7935 (min 0.5882) |
| fd_dml | 6 | private_ema/head | global_f1 0.5284 (min 0.3017) | local_f1 0.6086 (min 0.4035) | local_acc 0.8231 (min 0.6454) |
| fd_dml | 6 | private_ema/gp | global_f1 0.6257 (min 0.5166) | local_f1 0.6277 (min 0.4707) | local_acc 0.8009 (min 0.6299) |
| fd_dml | 6 | proxy/head | global_f1 0.7631 (min 0.7631) | local_f1 0.6628 (min 0.5576) | local_acc 0.7708 (min 0.6081) |
| fd_dml@e15 | 15 | private/head | global_f1 0.5707 (min 0.3425) | local_f1 0.6518 (min 0.4468) | local_acc 0.8273 (min 0.6441) |
| fd_dml@e15 | 15 | private/gp | global_f1 0.6960 (min 0.6049) | local_f1 0.6588 (min 0.5010) | local_acc 0.7883 (min 0.4319) |
| fd_dml@e15 | 15 | private_ema/head | global_f1 0.5691 (min 0.3600) | local_f1 0.6479 (min 0.4567) | local_acc 0.8298 (min 0.6533) |
| fd_dml@e15 | 15 | private_ema/gp | global_f1 0.6869 (min 0.6045) | local_f1 0.6595 (min 0.5260) | local_acc 0.8003 (min 0.6272) |
| fd_dml@e15 | 15 | proxy/head | global_f1 0.7652 (min 0.7652) | local_f1 0.6713 (min 0.5526) | local_acc 0.7441 (min 0.4584) |
| fd_dml_avgm | 6 | private/head | global_f1 0.5486 (min 0.3346) | local_f1 0.6306 (min 0.4335) | local_acc 0.8271 (min 0.6465) |
| fd_dml_avgm | 6 | private/gp | global_f1 0.6673 (min 0.5704) | local_f1 0.6416 (min 0.4829) | local_acc 0.7955 (min 0.6005) |
| fd_dml_avgm | 6 | private_ema/head | global_f1 0.5310 (min 0.3066) | local_f1 0.6103 (min 0.4061) | local_acc 0.8232 (min 0.6458) |
| fd_dml_avgm | 6 | private_ema/gp | global_f1 0.6307 (min 0.5304) | local_f1 0.6291 (min 0.4788) | local_acc 0.7981 (min 0.6314) |
| fd_dml_avgm | 6 | proxy/head | global_f1 0.0537 (min 0.0537) | local_f1 0.0477 (min 0.0242) | local_acc 0.0994 (min 0.0248) |
| fd_dml_hkd | 6 | private/head | global_f1 0.5553 (min 0.3578) | local_f1 0.6370 (min 0.4572) | local_acc 0.8282 (min 0.6552) |
| fd_dml_hkd | 6 | private/gp | global_f1 0.6710 (min 0.5882) | local_f1 0.6425 (min 0.5163) | local_acc 0.7967 (min 0.6094) |
| fd_dml_hkd | 6 | private_ema/head | global_f1 0.5362 (min 0.3105) | local_f1 0.6159 (min 0.4039) | local_acc 0.8294 (min 0.6548) |
| fd_dml_hkd | 6 | private_ema/gp | global_f1 0.6394 (min 0.5099) | local_f1 0.6328 (min 0.4622) | local_acc 0.8052 (min 0.6365) |
| fd_dml_hkd | 6 | proxy/head | global_f1 0.7661 (min 0.7661) | local_f1 0.6599 (min 0.5478) | local_acc 0.7835 (min 0.6536) |
| fd_dml_m4b | 6 | private/head | global_f1 0.5864 (min 0.3669) | local_f1 0.6735 (min 0.4803) | local_acc 0.8821 (min 0.7698) |
| fd_dml_m4b | 6 | private/gp | global_f1 0.7010 (min 0.6167) | local_f1 0.6827 (min 0.5319) | local_acc 0.8541 (min 0.7472) |
| fd_dml_m4b | 6 | private_ema/head | global_f1 0.5695 (min 0.3501) | local_f1 0.6541 (min 0.4664) | local_acc 0.8770 (min 0.7640) |
| fd_dml_m4b | 6 | private_ema/gp | global_f1 0.6711 (min 0.5587) | local_f1 0.6730 (min 0.5359) | local_acc 0.8543 (min 0.7561) |
| fd_dml_m4b | 6 | proxy/head | global_f1 0.8146 (min 0.8146) | local_f1 0.7119 (min 0.6073) | local_acc 0.8416 (min 0.6866) |
| fd_dml_m4b@e15 | 15 | private/head | global_f1 0.6088 (min 0.4315) | local_f1 0.6958 (min 0.5428) | local_acc 0.8907 (min 0.7788) |
| fd_dml_m4b@e15 | 15 | private/gp | global_f1 0.7323 (min 0.6286) | local_f1 0.7078 (min 0.5981) | local_acc 0.8667 (min 0.7558) |
| fd_dml_m4b@e15 | 15 | private_ema/head | global_f1 0.6048 (min 0.4037) | local_f1 0.6894 (min 0.5160) | local_acc 0.8878 (min 0.7772) |
| fd_dml_m4b@e15 | 15 | private_ema/gp | global_f1 0.7222 (min 0.6456) | local_f1 0.7038 (min 0.5997) | local_acc 0.8652 (min 0.7644) |
| fd_dml_m4b@e15 | 15 | proxy/head | global_f1 0.8510 (min 0.8510) | local_f1 0.7549 (min 0.6436) | local_acc 0.8610 (min 0.7424) |
| fd_dml_m4b_q4 | 6 | private/head | global_f1 0.5859 (min 0.3643) | local_f1 0.6734 (min 0.4796) | local_acc 0.8817 (min 0.7580) |
| fd_dml_m4b_q4 | 6 | private/gp | global_f1 0.7010 (min 0.6158) | local_f1 0.6825 (min 0.5344) | local_acc 0.8536 (min 0.7490) |
| fd_dml_m4b_q4 | 6 | private_ema/head | global_f1 0.5694 (min 0.3502) | local_f1 0.6538 (min 0.4670) | local_acc 0.8761 (min 0.7576) |
| fd_dml_m4b_q4 | 6 | private_ema/gp | global_f1 0.6707 (min 0.5637) | local_f1 0.6731 (min 0.5377) | local_acc 0.8534 (min 0.7500) |
| fd_dml_m4b_q4 | 6 | proxy/head | global_f1 0.8157 (min 0.8157) | local_f1 0.7136 (min 0.6159) | local_acc 0.8449 (min 0.6783) |
| fd_dml_m4b_sub4_q4 | 6 | private/head | global_f1 0.6015 (min 0.4026) | local_f1 0.6866 (min 0.5147) | local_acc 0.8860 (min 0.7758) |
| fd_dml_m4b_sub4_q4 | 6 | private/gp | global_f1 0.7319 (min 0.6367) | local_f1 0.6958 (min 0.5827) | local_acc 0.8520 (min 0.7023) |
| fd_dml_m4b_sub4_q4 | 6 | private_ema/head | global_f1 0.5857 (min 0.3745) | local_f1 0.6670 (min 0.4849) | local_acc 0.8796 (min 0.7652) |
| fd_dml_m4b_sub4_q4 | 6 | private_ema/gp | global_f1 0.7087 (min 0.6259) | local_f1 0.6866 (min 0.5730) | local_acc 0.8552 (min 0.7511) |
| fd_dml_m4b_sub4_q4 | 6 | proxy/head | global_f1 0.8485 (min 0.8485) | local_f1 0.7497 (min 0.6433) | local_acc 0.8672 (min 0.7181) |
| fd_dml_ntd | 6 | private/head | global_f1 0.5302 (min 0.2996) | local_f1 0.6120 (min 0.4049) | local_acc 0.8168 (min 0.6428) |
| fd_dml_ntd | 6 | private/gp | global_f1 0.6236 (min 0.3996) | local_f1 0.6099 (min 0.3779) | local_acc 0.7678 (min 0.5283) |
| fd_dml_ntd | 6 | private_ema/head | global_f1 0.5169 (min 0.2977) | local_f1 0.5961 (min 0.4020) | local_acc 0.8220 (min 0.6478) |
| fd_dml_ntd | 6 | private_ema/gp | global_f1 0.6035 (min 0.4083) | local_f1 0.6152 (min 0.4146) | local_acc 0.8058 (min 0.6318) |
| fd_dml_ntd | 6 | proxy/head | global_f1 0.7645 (min 0.7645) | local_f1 0.6594 (min 0.5534) | local_acc 0.7929 (min 0.6409) |
| fd_dml_prox | 6 | private/head | global_f1 0.5356 (min 0.3106) | local_f1 0.6182 (min 0.4173) | local_acc 0.8333 (min 0.6533) |
| fd_dml_prox | 6 | private/gp | global_f1 0.6527 (min 0.4409) | local_f1 0.5985 (min 0.3413) | local_acc 0.7271 (min 0.2427) |
| fd_dml_prox | 6 | private_ema/head | global_f1 0.5163 (min 0.2976) | local_f1 0.5963 (min 0.4009) | local_acc 0.8274 (min 0.6495) |
| fd_dml_prox | 6 | private_ema/gp | global_f1 0.6269 (min 0.4274) | local_f1 0.6038 (min 0.3657) | local_acc 0.7719 (min 0.4336) |
| fd_dml_prox | 6 | proxy/head | global_f1 0.7624 (min 0.7624) | local_f1 0.6394 (min 0.5181) | local_acc 0.7887 (min 0.6647) |
| fd_dml_q4 | 6 | private/head | global_f1 0.5424 (min 0.3042) | local_f1 0.6252 (min 0.4095) | local_acc 0.8241 (min 0.6468) |
| fd_dml_q4 | 6 | private/gp | global_f1 0.6613 (min 0.5774) | local_f1 0.6334 (min 0.4722) | local_acc 0.7888 (min 0.5788) |
| fd_dml_q4 | 6 | private_ema/head | global_f1 0.5272 (min 0.2984) | local_f1 0.6074 (min 0.3999) | local_acc 0.8216 (min 0.6468) |
| fd_dml_q4 | 6 | private_ema/gp | global_f1 0.6243 (min 0.5162) | local_f1 0.6259 (min 0.4676) | local_acc 0.7971 (min 0.6307) |
| fd_dml_q4 | 6 | proxy/head | global_f1 0.7651 (min 0.7651) | local_f1 0.6634 (min 0.5562) | local_acc 0.7773 (min 0.6384) |
| fd_dml_sub16_q4 | 6 | private/head | global_f1 0.5886 (min 0.3722) | local_f1 0.6697 (min 0.4717) | local_acc 0.8556 (min 0.6734) |
| fd_dml_sub16_q4 | 6 | private/gp | global_f1 0.7139 (min 0.5563) | local_f1 0.6531 (min 0.4314) | local_acc 0.7648 (min 0.2880) |
| fd_dml_sub16_q4 | 6 | private_ema/head | global_f1 0.5604 (min 0.3300) | local_f1 0.6396 (min 0.4264) | local_acc 0.8450 (min 0.6672) |
| fd_dml_sub16_q4 | 6 | private_ema/gp | global_f1 0.6886 (min 0.5418) | local_f1 0.6515 (min 0.4395) | local_acc 0.7975 (min 0.4308) |
| fd_dml_sub16_q4 | 6 | proxy/head | global_f1 0.8270 (min 0.8270) | local_f1 0.7262 (min 0.6149) | local_acc 0.8462 (min 0.6703) |
| fd_dml_sub4 | 6 | private/head | global_f1 0.5641 (min 0.3294) | local_f1 0.6456 (min 0.4321) | local_acc 0.8388 (min 0.6622) |
| fd_dml_sub4 | 6 | private/gp | global_f1 0.7004 (min 0.6321) | local_f1 0.6558 (min 0.5269) | local_acc 0.7961 (min 0.5594) |
| fd_dml_sub4 | 6 | private_ema/head | global_f1 0.5425 (min 0.3104) | local_f1 0.6226 (min 0.4124) | local_acc 0.8330 (min 0.6583) |
| fd_dml_sub4 | 6 | private_ema/gp | global_f1 0.6643 (min 0.5770) | local_f1 0.6463 (min 0.5020) | local_acc 0.8055 (min 0.6297) |
| fd_dml_sub4 | 6 | proxy/head | global_f1 0.7891 (min 0.7891) | local_f1 0.6888 (min 0.5690) | local_acc 0.7945 (min 0.6420) |
| fd_dml_sub4@e15 | 15 | private/head | global_f1 0.5908 (min 0.3862) | local_f1 0.6716 (min 0.4886) | local_acc 0.8427 (min 0.6655) |
| fd_dml_sub4@e15 | 15 | private/gp | global_f1 0.7211 (min 0.6333) | local_f1 0.6742 (min 0.5272) | local_acc 0.7899 (min 0.3926) |
| fd_dml_sub4@e15 | 15 | private_ema/head | global_f1 0.5826 (min 0.3664) | local_f1 0.6605 (min 0.4537) | local_acc 0.8388 (min 0.6616) |
| fd_dml_sub4@e15 | 15 | private_ema/gp | global_f1 0.7120 (min 0.6249) | local_f1 0.6685 (min 0.5236) | local_acc 0.8012 (min 0.5759) |
| fd_dml_sub4@e15 | 15 | proxy/head | global_f1 0.7926 (min 0.7926) | local_f1 0.6968 (min 0.5714) | local_acc 0.7634 (min 0.4902) |
| fd_dml_sub4_prox | 6 | private/head | global_f1 0.5629 (min 0.3419) | local_f1 0.6437 (min 0.4417) | local_acc 0.8398 (min 0.6506) |
| fd_dml_sub4_prox | 6 | private/gp | global_f1 0.6848 (min 0.4780) | local_f1 0.6266 (min 0.3708) | local_acc 0.7497 (min 0.2131) |
| fd_dml_sub4_prox | 6 | private_ema/head | global_f1 0.5443 (min 0.3324) | local_f1 0.6231 (min 0.4350) | local_acc 0.8355 (min 0.6537) |
| fd_dml_sub4_prox | 6 | private_ema/gp | global_f1 0.6662 (min 0.4944) | local_f1 0.6317 (min 0.4061) | local_acc 0.7869 (min 0.3970) |
| fd_dml_sub4_prox | 6 | proxy/head | global_f1 0.7806 (min 0.7806) | local_f1 0.6706 (min 0.5316) | local_acc 0.7860 (min 0.6216) |
| fd_dml_sub4_q4 | 6 | private/head | global_f1 0.5624 (min 0.3408) | local_f1 0.6436 (min 0.4444) | local_acc 0.8346 (min 0.6592) |
| fd_dml_sub4_q4 | 6 | private/gp | global_f1 0.6969 (min 0.6252) | local_f1 0.6523 (min 0.5189) | local_acc 0.7888 (min 0.5441) |
| fd_dml_sub4_q4 | 6 | private_ema/head | global_f1 0.5419 (min 0.3185) | local_f1 0.6221 (min 0.4167) | local_acc 0.8318 (min 0.6578) |
| fd_dml_sub4_q4 | 6 | private_ema/gp | global_f1 0.6634 (min 0.5733) | local_f1 0.6448 (min 0.4947) | local_acc 0.8038 (min 0.6315) |
| fd_dml_sub4_q4 | 6 | proxy/head | global_f1 0.7863 (min 0.7863) | local_f1 0.6868 (min 0.5666) | local_acc 0.7898 (min 0.6268) |
| fd_dml_swappos | 6 | private/head | global_f1 0.5841 (min 0.4077) | local_f1 0.6712 (min 0.5203) | local_acc 0.8765 (min 0.7587) |
| fd_dml_swappos | 6 | private/gp | global_f1 0.6985 (min 0.6131) | local_f1 0.6807 (min 0.5564) | local_acc 0.8454 (min 0.7485) |
| fd_dml_swappos | 6 | private_ema/head | global_f1 0.5657 (min 0.3746) | local_f1 0.6483 (min 0.4886) | local_acc 0.8689 (min 0.7534) |
| fd_dml_swappos | 6 | private_ema/gp | global_f1 0.6641 (min 0.5529) | local_f1 0.6657 (min 0.5545) | local_acc 0.8421 (min 0.6693) |
| fd_dml_swappos | 6 | proxy/head | global_f1 0.8121 (min 0.8121) | local_f1 0.7119 (min 0.5993) | local_acc 0.8504 (min 0.6341) |
| fd_fedavg | 6 | proxy/head | global_f1 0.7724 (min 0.7724) | local_f1 0.6565 (min 0.5173) | local_acc 0.7651 (min 0.5666) |
| fd_fedavg@e15 | 15 | proxy/head | global_f1 0.7733 (min 0.7733) | local_f1 0.6556 (min 0.4974) | local_acc 0.7429 (min 0.4387) |
| fd_fedavg_sub16_q4 | 6 | proxy/head | global_f1 0.8527 (min 0.8527) | local_f1 0.7406 (min 0.5814) | local_acc 0.8269 (min 0.6493) |
| fd_fedavg_sub4 | 6 | proxy/head | global_f1 0.8059 (min 0.8059) | local_f1 0.6876 (min 0.5199) | local_acc 0.7769 (min 0.5222) |
| fd_fedavg_sub4@e15 | 15 | proxy/head | global_f1 0.8142 (min 0.8142) | local_f1 0.7017 (min 0.5241) | local_acc 0.7795 (min 0.5102) |
| fd_fedavg_sub4_prox | 6 | proxy/head | global_f1 0.7792 (min 0.7792) | local_f1 0.6593 (min 0.5209) | local_acc 0.7681 (min 0.5473) |
| fd_fedavg_sub4_q4 | 6 | proxy/head | global_f1 0.7991 (min 0.7991) | local_f1 0.6834 (min 0.5235) | local_acc 0.7719 (min 0.5116) |
| fd_gaussian | 6 | private/head | global_f1 0.5479 (min 0.3922) | local_f1 0.6234 (min 0.4680) | local_acc 0.8014 (min 0.6298) |
| fd_gaussian | 6 | private/gp | global_f1 0.5882 (min 0.4583) | local_f1 0.6143 (min 0.4542) | local_acc 0.7896 (min 0.6167) |
| fd_gaussian | 6 | private_ema/head | global_f1 0.5695 (min 0.4405) | local_f1 0.6321 (min 0.5172) | local_acc 0.7988 (min 0.6514) |
| fd_gaussian | 6 | private_ema/gp | global_f1 0.6126 (min 0.4919) | local_f1 0.6190 (min 0.4930) | local_acc 0.7795 (min 0.6350) |
| fd_gaussian_m4b | 6 | private/head | global_f1 0.6019 (min 0.4094) | local_f1 0.6859 (min 0.5623) | local_acc 0.8817 (min 0.7532) |
| fd_gaussian_m4b | 6 | private/gp | global_f1 0.6553 (min 0.4899) | local_f1 0.6786 (min 0.5583) | local_acc 0.8662 (min 0.7528) |
| fd_gaussian_m4b | 6 | private_ema/head | global_f1 0.5976 (min 0.4266) | local_f1 0.6752 (min 0.5712) | local_acc 0.8692 (min 0.7529) |
| fd_gaussian_m4b | 6 | private_ema/gp | global_f1 0.6452 (min 0.5034) | local_f1 0.6669 (min 0.5560) | local_acc 0.8480 (min 0.7524) |
| fd_gaussian_swappos | 6 | private/head | global_f1 0.5907 (min 0.4246) | local_f1 0.6741 (min 0.5390) | local_acc 0.8768 (min 0.7560) |
| fd_gaussian_swappos | 6 | private/gp | global_f1 0.6404 (min 0.5130) | local_f1 0.6700 (min 0.5251) | local_acc 0.8609 (min 0.7539) |
| fd_gaussian_swappos | 6 | private_ema/head | global_f1 0.6205 (min 0.4674) | local_f1 0.6874 (min 0.5799) | local_acc 0.8741 (min 0.7664) |
| fd_gaussian_swappos | 6 | private_ema/gp | global_f1 0.6768 (min 0.5602) | local_f1 0.6619 (min 0.5374) | local_acc 0.8446 (min 0.7354) |

[gates.json](gates.json) · [10 metrics](metrics-all.csv) · [per-class](per-class-f1.csv)
