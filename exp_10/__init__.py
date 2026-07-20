"""exp10_text_classification：五种掩码报告文本编码器三标签分类。"""

MODEL_NAMES = (
    "hashed_mean_encoder",
    "vocab_attention_encoder",
    "textcnn_encoder",
    "bigru_encoder",
    "transformer_encoder",
)

__all__ = ["MODEL_NAMES"]
