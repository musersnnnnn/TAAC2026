
## DIN的做法 ##


DIN输入的是query，seq_tokens_list，seq_padding_masks，query指的是给定当前物品和用户画像信息，能从历史序列里面拿到多少有用的信息（模型先有NS token和Seq token，每一条序列单独生成query，每个序列域生成一个query，把所有NS token展平，把seq token做mean pooling，拼起来之后过一个小FFN得到query token）query作为item_anchor，先把query扩展到序列的每一个位置，然后拼q，seq，q-seq，q*seq四种信息，然后过一个mlp，序列的每一个位置上的历史行为都会输出一个分数，然后softmax成权重，然后对序列中每一个历史行为向量做加权求和得到一个context，这个context是一条历史序列里“和当前商品最相关的信息摘要，四条序列分别做DIN，最终分数需要门控来控制，context和query拼接之后送到门控投影出来门控分数，门控分数做softmax之后，和每条序列的context做加权融合，得到融合后的DIN context，这个context和query拼接后送到输出投影层，然后加到HyFormer的输出上。





TAAC 2026 腾讯广告算法大赛｜广告 CVR 预估模型优化
基于官方 HyFormer / RankMixer baseline 进行单模优化。baseline 采用统一 token 化建模思路：将用户 / 商品非序列特征编码为 NS tokens，将四个行为域历史序列编码为 sequence tokens，通过 Query Generator、Cross Attention 与 RankMixer Block 完成非序列特征和序列行为的统一交互建模。
在深入分析 baseline 数据与模型链路后，围绕 dense 信息压缩、时间分布建模、候选 item 与历史行为交互不足等问题进行改进，最终线上 AUC 从复跑 baseline 的 0.805927 提升至 0.831174。
数据侧构造 raw user_dense 统计特征、当前样本时间特征、序列时间特征及 item-history match 特征，显式补充用户画像分布、时间上下文和候选 item 与历史兴趣之间的匹配信号。
模型侧引入 DIN target-aware sequence reader，以 item 表示作为 query 对四个行为域进行 attention pooling，并通过 domain gate 融合多域历史兴趣，再以零初始化 residual 方式注入 HyFormer 主干，增强 target-aware sequence interaction。
进一步采用多 dense token、online proxy AUC 选点、target-time / long-history 样本加权、pairwise AUC loss、label smoothing、logit L2、dense EMA 与 gated tail residual calibration，提高模型排序能力和线上稳定性。




基于官方 HyFormer / RankMixer baseline 进行单模优化，围绕用户画像、候选 item 与多域历史行为序列交互建模，最终线上 AUC 达到 0.831174，较复跑 baseline 0.812927 提升约 +0.0182。负责构建 raw user_dense 统计、当前时间 / 序列时间、item-history match 等特征，补充 dense 重尾分布与候选 item-历史兴趣显式交互信息；在模型侧引入 DIN target-aware sequence reader，以 item 表示作为 query 对四个行为域进行 attention pooling，并通过 residual 融合进 HyFormer 主干；训练侧设计 online proxy AUC 选点、target-time / long-history 样本加权、pairwise AUC loss、label smoothing、logit L2、dense EMA 与 gated tail residual calibration，提升模型排序能力和线上稳定性。







这个项目我不是直接重写模型，而是在官方 baseline 的设计上做针对性增强。官方 baseline 的核心是 HyFormer / RankMixer：先把用户和商品的非序列特征通过 NS Tokenizer 转成 tokens，再把四个 domain 的历史行为序列转成 sequence tokens；然后用 Query Generator 为每条序列生成 query，通过 Cross Attention 从历史序列里读信息，最后用 RankMixer 做 token 之间的低成本交互。
我主要发现 baseline 有几个信息缺口：第一，user_dense 信息非常重要，但压成少量 token 后容易丢失重尾统计；第二，当前样本时间和序列时间对线上分布有明显影响；第三，baseline 主要依赖模型自己学习候选 item 和历史行为的关系，缺少显式 item-history match 信号；第四，HyFormer 的 query 机制不够直接地以当前 item 为中心读取历史。
所以我在数据侧补了 dense statistics、当前时间、序列时间和 item-history match 特征；在模型侧加了 DIN 风格的 target-aware sequence reader，用当前 item 表示作为 query 去四个历史行为域里做 attention pooling，并通过 residual 方式融合回 HyFormer 主干。训练侧再用 online proxy AUC、样本加权、pairwise AUC loss、label smoothing、logit L2、dense EMA 和 tail calibration 提升稳定性。最终单模线上 AUC 达到 0.831174，相比复跑官方 baseline 提升约 0.0182。



