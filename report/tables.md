### Качество reward model (отложенная выборка = последний событие пользователя)

| метрика | значение |
|---|---|
| RMSE | 0.4087 |
| Spearman (RM vs true) | 0.1156 |
| RMSE baseline (среднее по item) | 0.3793 |
| Spearman baseline | 0.3761 |
| n | 84511 |

### Recall/NDCG/MAP и поведение top-10

| модель | Recall@10 | NDCG@10 | MAP@10 | RM-reward top1 | RM top1 (без маски) | Rewatch@10 | Coverage@10 | Entropy(top10) | Top-1 share |
|---|---|---|---|---|---|---|---|---|---|
| base (без RL) | 0.1129 | 0.0613 | 0.0454 | 0.5305 | 0.5294 | 0.0905 | 0.0053 | 0.2985 | 0.0940 |
| kion_completion | 0.1066 | 0.0564 | 0.0412 | 0.5416 | 0.5437 | 0.0866 | 0.0048 | 0.2959 | 0.0938 |
| kion_completion_nokl | 0.0833 | 0.0481 | 0.0373 | 0.5418 | 0.5437 | 0.0707 | 0.0039 | 0.2952 | 0.0935 |
| kion_completion_norep | 0.1104 | 0.0567 | 0.0402 | 0.5250 | 0.5235 | 0.0898 | 0.0059 | 0.2991 | 0.0929 |
| kion_completion_poppen | 0.1071 | 0.0541 | 0.0379 | 0.5152 | 0.5153 | 0.0906 | 0.0073 | 0.3160 | 0.0924 |
| kion_grpo | 0.0740 | 0.0447 | 0.0355 | 0.5399 | 0.5437 | 0.0654 | 0.0042 | 0.3001 | 0.0973 |

### Многошаговая генерация (5 шагов, контекст = train-история)

| модель | шаг | RM reward | повтор в контексте | logprob политики |
|---|---|---|---|---|
| base | step1 | 0.5233 | 0.0600 | -3.644 |
| base | step2 | 0.5183 | 0.1133 | -3.661 |
| base | step3 | 0.5172 | 0.1133 | -3.711 |
| base | step4 | 0.5176 | 0.1333 | -3.730 |
| base | step5 | 0.5183 | 0.1700 | -3.737 |
| kion_completion | step1 | 0.5250 | 0.0733 | -3.569 |
| kion_completion | step2 | 0.5263 | 0.1033 | -3.568 |
| kion_completion | step3 | 0.5258 | 0.1600 | -3.485 |
| kion_completion | step4 | 0.5273 | 0.1967 | -3.562 |
| kion_completion | step5 | 0.5291 | 0.2100 | -3.489 |
| base | step1 | 0.5206 | 0.0333 | -3.681 |
| base | step2 | 0.5190 | 0.1033 | -3.609 |
| base | step3 | 0.5169 | 0.1367 | -3.632 |
| base | step4 | 0.5183 | 0.1567 | -3.660 |
| base | step5 | 0.5191 | 0.1667 | -3.750 |
| kion_completion_nokl | step1 | 0.5373 | 0.0733 | -2.233 |
| kion_completion_nokl | step2 | 0.5359 | 0.4633 | -1.986 |
| kion_completion_nokl | step3 | 0.5352 | 0.5767 | -2.004 |
| kion_completion_nokl | step4 | 0.5368 | 0.6000 | -1.955 |
| kion_completion_nokl | step5 | 0.5370 | 0.6233 | -2.028 |
| base | step1 | 0.5202 | 0.0700 | -3.717 |
| base | step2 | 0.5192 | 0.0867 | -3.648 |
| base | step3 | 0.5178 | 0.1233 | -3.710 |
| base | step4 | 0.5210 | 0.1567 | -3.684 |
| base | step5 | 0.5200 | 0.1667 | -3.730 |
| kion_completion_norep | step1 | 0.5238 | 0.0467 | -3.814 |
| kion_completion_norep | step2 | 0.5206 | 0.1167 | -3.793 |
| kion_completion_norep | step3 | 0.5203 | 0.0600 | -3.798 |
| kion_completion_norep | step4 | 0.5214 | 0.1567 | -3.807 |
| kion_completion_norep | step5 | 0.5222 | 0.1167 | -3.808 |
| base | step1 | 0.5215 | 0.0533 | -3.719 |
| base | step2 | 0.5192 | 0.0900 | -3.731 |
| base | step3 | 0.5175 | 0.0967 | -3.668 |
| base | step4 | 0.5182 | 0.1633 | -3.724 |
| base | step5 | 0.5194 | 0.1767 | -3.609 |
| kion_completion_poppen | step1 | 0.5216 | 0.0200 | -3.882 |
| kion_completion_poppen | step2 | 0.5180 | 0.1067 | -3.836 |
| kion_completion_poppen | step3 | 0.5161 | 0.0900 | -3.871 |
| kion_completion_poppen | step4 | 0.5212 | 0.1167 | -3.891 |
| kion_completion_poppen | step5 | 0.5190 | 0.1167 | -3.873 |
| base | step1 | 0.5203 | 0.0533 | -3.685 |
| base | step2 | 0.5181 | 0.1000 | -3.682 |
| base | step3 | 0.5154 | 0.1000 | -3.723 |
| base | step4 | 0.5199 | 0.1400 | -3.708 |
| base | step5 | 0.5211 | 0.2100 | -3.705 |
| kion_grpo | step1 | 0.5451 | 0.1133 | -0.264 |
| kion_grpo | step2 | 0.5434 | 0.9567 | -0.098 |
| kion_grpo | step3 | 0.5433 | 0.9867 | -0.117 |
| kion_grpo | step4 | 0.5429 | 0.9700 | -0.240 |
| kion_grpo | step5 | 0.5435 | 0.9867 | -0.147 |

### Динамика RL (по логам rl_log.json)

| run | step | reward | kl | entropy | reward_top1_eval |
|---|---|---|---|---|---|
| kion_completion | 50 | 0.5263 | 0.0166 | 7.494 | - |
| kion_completion | 450 | 0.5224 | -0.0082 | 7.485 | - |
| kion_completion | 800 | 0.5252 | 0.0198 | 7.422 | 0.5412 |
| kion_completion_nokl | 50 | 0.5246 | 0.0109 | 7.482 | - |
| kion_completion_nokl | 450 | 0.5273 | 0.0363 | 7.350 | - |
| kion_completion_nokl | 800 | 0.5297 | 0.5506 | 6.182 | 0.5415 |
| kion_completion_norep | 50 | 0.5084 | 0.0136 | 7.498 | - |
| kion_completion_norep | 450 | 0.4944 | -0.0037 | 7.568 | - |
| kion_completion_norep | 800 | 0.5235 | 0.0153 | 7.608 | 0.5247 |
| kion_completion_poppen | 50 | 0.3112 | 0.0064 | 7.514 | - |
| kion_completion_poppen | 450 | 0.3214 | 0.0525 | 7.653 | - |
| kion_completion_poppen | 800 | 0.3218 | 0.0565 | 7.712 | 0.5147 |
| kion_grpo | 50 | 0.5255 | -0.0076 | 7.485 | - |
| kion_grpo | 450 | 0.5312 | 0.3930 | 6.403 | - |
| kion_grpo | 800 | 0.5439 | 3.5147 | 1.049 | 0.5395 |
