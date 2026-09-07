# TAAC 2026 初赛方案开源

本仓库开源的是我们参加 **TAAC 2026 腾讯广告算法大赛初赛** 时的最终单模方案。

最终提交版本为：

- `baseline_v9_dualhead_tail_calib`
- 线上 AUC：**0.831174**
- 我们复跑的官方 baseline AUC：**0.812927**
- 相对提升：约 **+0.0182 AUC**

遗憾的是，这个成绩最终没有进入复赛。但整个比赛过程中，我们围绕官方 baseline 做了比较系统的特征工程、模型结构、训练策略和验证方式探索。这个仓库主要希望记录我们认为有效的部分，以及一些容易踩坑的方向，供后来参考。

## 方案概览

最终方案整体仍然基于官方 baseline 的 HyFormer / RankMixer 主干，没有完全重写模型，也没有做大规模 scale up。

我们最后的主要思路是：

> 在 baseline 主干上补充缺失的数据分布信息和候选 item - 用户历史交互信息，再用 EMA、online proxy 和轻量 tail calibration 提升稳定性与尾部排序。

最终版主要包含：

- raw user_dense statistics
- 当前样本时间特征
- 序列时间特征
- item-history match / pair clean 特征
- DIN target-aware sequence reader
- 多 dense token
- online proxy 验证与样本加权
- light pairwise AUC loss
- label smoothing / logit L2
- dense EMA
- dual-head tail calibration

## 最终版本：baseline_v9_dualhead_tail_calib

## 主要改动

### 1. Raw user_dense statistics

`user_dense` 是本赛题里非常重要的一类特征，而且分布明显重尾。直接把 dense 特征压成少量 token，容易损失尾部分布信息。

最终版保留原始 `user_dense`，同时额外追加 32 维统计特征，包括：

- 总和、最大值、L2 norm
- 非零比例
- 非零均值
- 最大值占总和比例
- 大于 1k / 100k 的高值比例
- 按 chunk 切分后的 sum / max / nonzero ratio

这部分是后期最稳定的上分点之一。

### 2. 当前样本时间特征

测试集和时间分布高度相关，因此最终版加入了当前样本时间特征。

Sparse 侧加入：

- hour token
- weekday token
- weekday-hour cross token

Dense 侧加入：

- hour sin / cos
- weekday sin / cos
- 是否周一
- 是否 7~9 点目标时间段

这样模型可以直接感知当前请求所处的时间上下文。

### 3. 序列时间特征

除了当前样本时间，最终版也保留了序列行为的时间信息。

每条序列会构造：

- 序列长度比例
- log length
- 最近行为时间间隔
- 平均时间间隔
- 序列时间跨度
- 近 1 天 / 近 7 天行为强度

同时，序列 token 中也加入 time bucket embedding，用于表达历史行为距离当前样本的时间差。

### 4. Item-history match / pair clean

最终版包含候选 item 和用户历史行为之间的显式匹配统计，也就是 item-history match summaries。

对每条行为序列，我们统计候选 item sparse 特征和历史行为 sparse 特征之间的匹配情况，包括：

- 总匹配数
- 有匹配的历史事件数
- 是否存在匹配
- 第 1 个位置、前 10、前 50 的匹配比例
- 最近匹配强度
- 第一次 / 最后一次匹配位置
- 有匹配的 feature 数
- 匹配事件距离当前样本的最小 / 平均时间差
- 1 天 / 7 天 / 30 天内匹配比例
- 时间衰减后的匹配分数

同时忽略 `<=2` 的值，减少 padding、missing、小命名空间类别导致的伪匹配。

这部分可以理解为一种更克制的 pair 特征：不直接记忆 pair，而是补充候选 item 与用户历史兴趣之间的显式交互信号。

### 5. DIN target-aware sequence reader

最终版加入了 DIN 风格的 target-aware pooling。

大致流程：

1. 用 item 侧 token 聚合出 `item_anchor`；
2. 用 `item_anchor` 作为 query，对四条用户行为序列分别做 DIN attention；
3. 得到每条序列的 target-aware context；
4. 使用 domain gate 融合四条序列；
5. 通过 residual 方式加回 HyFormer 输出。

DIN residual 最后一层是零初始化的，因此训练初始等价于原模型，后续逐渐学习候选 item 与历史行为之间的交互。

### 6. 多 dense token

最终版使用：

```text
user_dense_tokens=5
item_dense_tokens=1
```

测试集中 item_dense 为空，但 user_dense 会被切成多个 dense token，减少过早压缩 dense 信息的问题。

### 7. Online proxy 验证与样本加权

官方 valid AUC 和线上分布并不完全一致。我们发现测试集更偏向目标时间段和长历史用户，因此构造了 `online_proxy_auc` 辅助选 checkpoint。

`online_proxy_auc` 综合考虑：

- target time AUC
- online long-history AUC
- all valid AUC
- hour AUC

训练时也使用：

```text
sample_weight_mode=online_proxy
```

对目标时间段和长历史样本做轻微加权。

### 8. Light pairwise AUC loss

最终版在 BCE 之外加入了轻量 pairwise AUC loss：

```text
rank_loss_weight=0.05
rank_loss_temperature=1.0
rank_loss_max_pairs=8192
```

这个 loss 用来强化排序能力，但权重较小，避免破坏主 BCE 训练。

### 9. Label smoothing 和 logit L2

为了缓解模型在高分尾部过度自信的问题，最终版加入：

```text
label_smoothing=0.003
logit_l2=0.0002
```

这两个正则都比较轻，主要用于稳定输出。

### 10. Dense EMA

最终版使用 dense 参数 EMA：

```text
dense_ema_decay=0.999
dense_ema_start_step=7248
```

只对 dense 参数做 EMA，不对 sparse embedding 做 EMA，因为 sparse embedding 仍然保留每轮 re-init 策略。

EMA 主要用于提升 checkpoint 稳定性。

### 11. Dual-head tail calibration

最终版加入 gated tail residual head，用于修正高分尾部排序。

形式上：

```text
base_logit = base_head(output)
tail_delta = gate(output, din_context, tail_meta) * residual_head(output, din_context, tail_meta)
final_logit = base_logit + scale * tail_delta
```

其中：

```text
tail_residual_scale=0.20
tail_aux_loss_weight=0.12
tail_residual_l2=0.0005
tail_high_score_quantile=0.80
```

这个模块不是为了重建一个新模型，而是在 baseline 已有排序能力上，对高分尾部做受控修正。



## 分数记录

| 版本 / 方向                              |     线上 AUC | 说明                                  |
| ---------------------------------------- | -----------: | ------------------------------------- |
| 官方 baseline 复跑                       |     0.812927 | 起点                                  |
| 时间 / token 早期优化                    |     0.823186 | 时间信息和 token 处理明显有效         |
| DIN + pair clean + timeauc               |     0.827027 | 候选 item 与历史交互开始带来收益      |
| dense stats + EMA                        |     0.829781 | raw dense statistics 是后期稳定上分点 |
| dual-head stable                         |     0.830541 | tail calibration 稳定带来增益         |
| 最终版 `baseline_v9_dualhead_tail_calib` | **0.831174** | 最终最好单模                          |

这些分数不是严格单变量消融，因为不同实验之间可能同时包含多个改动，也存在随机种子和 checkpoint 选择影响，仅供方向参考。

## 经验总结

我们认为最终版有效，主要来自以下几点：

1. `user_dense` 重尾信息非常重要，raw dense statistics 能补回 dense token 压缩损失的信息；
2. 当前样本时间和序列时间都对线上分布有帮助；
3. 候选 item 与用户历史行为之间的显式匹配信息是有效的；
4. DIN residual 可以补充 target-aware sequence interaction；
5. 只看全量 valid AUC 容易误判，online proxy 对选点有帮助；
6. pairwise AUC loss、label smoothing、logit L2 能轻微改善排序和过置信；
7. dense EMA 能提升 checkpoint 稳定性；
8. tail calibration 对高分尾部排序有帮助，但必须克制，过强容易过拟合。

## 失败经验

我们也尝试过一些更激进的方向，包括：

- 更大的模型
- 更长序列
- MoE
- stage2 tail finetune
- sparse lock
- 更复杂的 tail meta
- 更强的 target reweight
- fulltrain tiny valid

很多版本在 valid 上看起来不错，但线上并不稳定，甚至明显掉分。

因此最终版本选择了相对克制的路线：不大幅 scale up 主干，而是补充信息、改进验证、轻量校准。

