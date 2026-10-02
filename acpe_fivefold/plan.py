"""先完整五折比较机制，再对完成结果进行单因素深度扩展。"""
R51 = {'descriptor': 'no_current', 'position_warmup': 8}
BREADTH = [
    ('original_pe', {'position': 'original_pe'}, '共同对照：标准位置编码'),
    ('r051', R51, '共同对照：删除当前特征输入、8轮位置注入warmup'),
    ('original_structured_training', {'position': 'original_pe', 'structured_training': True}, '严重连续缺失训练的匹配Original PE对照'),
    ('r051_structured_training', {**R51, 'structured_training': True}, '在原始序列删除后重采样训练，缩小严重缺失分布差异'),
    ('gap_confidence', {**R51, 'gap_confidence': True}, '相邻观测间距增大时收缩不可靠的内容形变'),
    ('old_acpe', {}, '共同对照：保留原始ACPE实现，识别历史改动效应'),
    ('valid_context', {'descriptor': 'valid_neighbors', 'position_warmup': 8}, '补齐边界掩码后，保留完整描述符'),
    ('r051_no_warmup', {'descriptor': 'no_current'}, '区分描述符变化与训练warmup'),
    ('anchor_only', {**R51, 'position': 'anchored'}, '固定c=r，检验是否确实需要内容变形'),
    ('compact_descriptor', {**R51, 'compact_descriptor': True}, '实际去除零向量及其归一化槽位，保留5组+1'),
    ('magnitude_descriptor', {**R51, 'context_form': 'magnitude'}, '仅保留两侧幅度与间距，减少符号噪声'),
    ('direction_descriptor', {**R51, 'context_form': 'direction'}, '仅保留有符号转变与间距'),
    ('gap_scaled_transition', {**R51, 'context_form': 'velocity'}, '用相对采集间距归一化视觉转变，解耦缺失与内容变化'),
    ('coherence_confidence', {**R51, 'eta_confidence': True}, '两侧转变不一致时降低局部形变置信度'),
    ('smooth_context', {**R51, 'context_form': 'smooth'}, '在局部平滑特征上估计间距，降低孤立异常帧影响'),
    ('global_context', {**R51, 'context_form': 'global'}, '用检查级均值补充局部描述，不恢复当前帧特征'),
    ('identity_warp_init', {**R51, 'zero_warp_init': True}, '从c=r启动，检验初始化稳定性'),
    ('learned_route_scale', {**R51, 'learned_scale': .1}, '绝对通道和相对注意力头分别学习小尺度位置残差'),
    ('content_route_gate', {**R51, 'content_gate': True}, '按检查内容选择位置注入强度'),
    ('original_plus_context', {**R51, 'base_original': True, 'learned_scale': .1}, '保留标准顺序先验，小残差补充采集上下文'),
    ('original_plus_correction', {**R51, 'base_original': True, 'correction_only': True, 'learned_scale': .1}, '仅注入相对锚定状态的上下文修正，避免重复位置项'),
    ('absolute_only', {**R51, 'route': 'absolute'}, '定位绝对注入是否承担主要分类作用'),
    ('relative_only', {**R51, 'route': 'relative'}, '减少直接扰动实例表征，仅调节关系'),
    ('relative_last', {**R51, 'placement': 'relative_last'}, '先建模内容再加入上下文关系偏置'),
    ('late_both', {**R51, 'placement': 'late_both'}, '将两条位置路径移至末层'),
    ('label_position', {**R51, 'label_position': True}, '检验上下文位置能否通过标签聚合直接服务分类'),
    ('original_no_lqd', {'position': 'original_pe', 'lqd_mode': 'off'}, '关闭LQD的匹配Original PE对照'),
    ('r051_no_lqd', {**R51, 'lqd_mode': 'off'}, '检验辅助损失对位置训练的影响；不当作纯ACPE结构收益'),
    ('original_visual_aux', {'position': 'original_pe', 'image_weight': .25}, '视觉辅助监督的匹配Original PE对照'),
    ('r051_visual_aux', {**R51, 'image_weight': .25}, '为视觉位置分支提供额外分类梯度'),
    ('detached_position_input', {**R51, 'detach_position_input': True}, '切断位置估计对视觉特征的反向扰动'),
]

# 后续按已完成五折结果的前三个结构候选循环扩展，全部保持单因素变化。
DEPTH = [
    {'warp_alpha': .5}, {'warp_alpha': 1.}, {'position_lr_multiplier': 3.},
    {'relative_scale': .25}, {'absolute_scale': .25}, {'position_warmup': 0},
    {'position_warmup': 4}, {'position_warmup': 12}, {'placement': 'relative_first'},
    {'learned_scale': .01}, {'learned_scale': .3}, {'eta_confidence': True},
    {'context_form': 'velocity'}, {'context_form': 'smooth'},
    {'compact_descriptor': True}, {'zero_warp_init': True},
    {'detach_position_input': True}, {'coordinate_penalty': .01},
    {'structured_training': True}, {'gap_confidence': True},
]
MAX_CANDIDATES = 48
