# Unified-Multimodal-Brain-Decoding-via-Cross-Subject-Soft-ROI-Fusion
- `neuro_informed_attn_test.py`：核心的神经科学先验 fMRI 编码器，结合体素坐标编码、多图谱 soft-ROI 先验和注意力聚合机制，将变长的 fMRI 体素信号转换为固定长度的 fMRI token 表征。
- perceiver.py: Perceiver-based resampling module for compressing variable-length input features into fixed latent tokens.
