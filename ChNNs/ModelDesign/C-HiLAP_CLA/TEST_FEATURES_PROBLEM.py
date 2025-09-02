import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import LibriSpeechDataset, extract_features
from chaos_features import ChaoticFeatureExtractor

# 配置参数
TEST_CONFIG = {
    "sample_audio_dir": "P:/PycharmProjects/pythonProject1/dataset/Small_Libri",  # 存放少量测试音频的目录
    "segment_length": 3.0,
    "sampling_rate": 16000,
    "n_mfcc": 40,
    "embedding_dim": 3,
    "delay": 1,
    "mlsa_scales": [1, 2, 4],
    "n_lyapunov_exponents": 2,
    "rqa_threshold": None
}


def test_feature_basic_properties():
    """测试特征基本属性（维度、异常值等）"""
    print("=== 测试特征基本属性 ===")

    # 加载单个音频文件
    audio_files = librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])
    if not audio_files:
        print("Error: 未找到测试音频文件")
        return
    test_audio = audio_files[0]
    audio, sr = librosa.load(test_audio, sr=TEST_CONFIG["sampling_rate"])

    # 测试MFCC特征
    mfcc_features = extract_features(audio, TEST_CONFIG["sampling_rate"])
    print(f"MFCC特征形状: {mfcc_features.shape}")  # 应为 (时间步, 120) 40*3
    print(f"MFCC均值: {np.mean(mfcc_features):.4f}, 标准差: {np.std(mfcc_features):.4f}")
    print(f"MFCC含零比例: {np.mean(mfcc_features == 0):.4f}")
    print(f"MFCC含NaN: {np.isnan(mfcc_features).any()}, 含无穷大: {np.isinf(mfcc_features).any()}")

    # 测试混沌特征
    chaotic_extractor = ChaoticFeatureExtractor(
        embedding_dim=TEST_CONFIG["embedding_dim"],
        delay=TEST_CONFIG["delay"],
        scales=TEST_CONFIG["mlsa_scales"],
        n_lyapunov_exponents=TEST_CONFIG["n_lyapunov_exponents"],
        rqa_threshold=TEST_CONFIG["rqa_threshold"]
    )
    chaotic_features = chaotic_extractor.extract(audio)
    print(f"\n混沌特征形状: {chaotic_features.shape}")  # 应为 (MLSA特征数 + RQA特征数,)
    print(f"混沌特征均值: {np.mean(chaotic_features):.4f}, 标准差: {np.std(chaotic_features):.4f}")
    print(f"混沌特征含NaN: {np.isnan(chaotic_features).any()}, 含无穷大: {np.isinf(chaotic_features).any()}")

    # 可视化MFCC特征
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_features.T[:40], sr=sr, x_axis="time")  # 只显示基础MFCC
    plt.colorbar()
    plt.title("MFCC特征可视化")
    plt.tight_layout()
    plt.savefig("mfcc_visualization.png")
    print("MFCC可视化已保存为 mfcc_visualization.png")


def test_speaker_discrimination():
    """测试特征对不同说话人的区分能力"""
    print("\n=== 测试说话人区分能力 ===")

    # 按说话人分组加载音频（假设目录结构为 speaker_id/utt/*.flac）
    speaker_dirs = [d for d in os.listdir(TEST_CONFIG["sample_audio_dir"])
                    if os.path.isdir(os.path.join(TEST_CONFIG["sample_audio_dir"], d))]
    if len(speaker_dirs) < 2:
        print("Error: 需要至少2个不同说话人的音频进行测试")
        return

    # 提取每个说话人的特征
    speaker_features = {}
    chaotic_extractor = ChaoticFeatureExtractor(
        embedding_dim=TEST_CONFIG["embedding_dim"],
        delay=TEST_CONFIG["delay"],
        scales=TEST_CONFIG["mlsa_scales"],
        n_lyapunov_exponents=TEST_CONFIG["n_lyapunov_exponents"]
    )

    for speaker in speaker_dirs[:2]:  # 取前2个说话人
        spk_dir = os.path.join(TEST_CONFIG["sample_audio_dir"], speaker)
        audio_files = librosa.util.find_files(spk_dir, ext=["flac", "wav"])[:3]  # 每个说话人取3个音频
        if not audio_files:
            continue

        mfcc_list = []
        chaotic_list = []
        for f in audio_files:
            audio, _ = librosa.load(f, sr=TEST_CONFIG["sampling_rate"])
            mfcc = extract_features(audio)
            chaotic = chaotic_extractor.extract(audio)
            mfcc_list.append(np.mean(mfcc, axis=0))  # 时间维度平均
            chaotic_list.append(chaotic)

        speaker_features[speaker] = {
            "mfcc": np.array(mfcc_list),
            "chaotic": np.array(chaotic_list)
        }
        print(f"说话人 {speaker} 提取了 {len(audio_files)} 个样本特征")

    # 计算同一说话人内部的相似度
    spk1, spk2 = speaker_dirs[:2]
    for feat_type in ["mfcc", "chaotic"]:
        # 同一说话人相似度
        spk1_sim = np.mean(cosine_similarity(speaker_features[spk1][feat_type]))
        spk2_sim = np.mean(cosine_similarity(speaker_features[spk2][feat_type]))
        # 不同说话人相似度
        cross_sim = np.mean(cosine_similarity(
            speaker_features[spk1][feat_type],
            speaker_features[spk2][feat_type]
        ))

        print(f"\n{feat_type}特征 - 同一说话人平均相似度: {np.mean([spk1_sim, spk2_sim]):.4f}")
        print(f"{feat_type}特征 - 不同说话人平均相似度: {cross_sim:.4f}")
        if np.mean([spk1_sim, spk2_sim]) > cross_sim:
            print(f"{feat_type}特征具有一定区分能力")
        else:
            print(f"警告: {feat_type}特征区分能力不足")

    # t-SNE可视化特征分布
    all_features = []
    all_labels = []
    for i, speaker in enumerate(speaker_dirs[:2]):
        all_features.extend(speaker_features[speaker]["mfcc"])
        all_labels.extend([i] * len(speaker_features[speaker]["mfcc"]))

    tsne = TSNE(n_components=2, perplexity=3)
    tsne_result = tsne.fit_transform(np.array(all_features))
    plt.figure(figsize=(8, 6))
    for label in set(all_labels):
        mask = np.array(all_labels) == label
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=speaker_dirs[label])
    plt.legend()
    plt.title("MFCC特征t-SNE分布")
    plt.savefig("mfcc_tsne.png")
    print("MFCC特征分布可视化已保存为 mfcc_tsne.png")


def test_cache_consistency():
    """测试缓存特征与实时提取特征的一致性"""
    print("\n=== 测试缓存一致性 ===")
    if not TEST_CONFIG["sample_audio_dir"]:
        return

    # 创建临时数据集
    dataset = LibriSpeechDataset(
        audio_files=librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])[:2],
        root_dir=TEST_CONFIG["sample_audio_dir"],
        segment_length=TEST_CONFIG["segment_length"],
        sampling_rate=TEST_CONFIG["sampling_rate"],
        cache_dir="temp_cache"
    )

    # 第一次加载（无缓存）
    item1 = dataset[0]
    # 第二次加载（有缓存）
    item2 = dataset[0]

    # 比较特征是否一致
    mfcc_diff = np.mean(np.abs(item1["audio"].numpy() - item2["audio"].numpy()))
    print(f"缓存前后MFCC特征平均差异: {mfcc_diff:.6f}")
    if mfcc_diff < 1e-6:
        print("缓存一致性检查通过")
    else:
        print("警告: 缓存特征与原始特征不一致")


def test_silence_removal():
    """测试静音去除是否过度"""
    print("\n=== 测试静音去除 ===")
    audio_files = librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])
    if not audio_files:
        return

    audio, sr = librosa.load(audio_files[0], sr=TEST_CONFIG["sampling_rate"])
    original_length = len(audio)

    # 模拟数据集中的静音去除过程
    dataset = LibriSpeechDataset(
        audio_files=[audio_files[0]],
        root_dir=TEST_CONFIG["sample_audio_dir"],
        segment_length=TEST_CONFIG["segment_length"],
        sampling_rate=TEST_CONFIG["sampling_rate"]
    )
    processed_audio = dataset._process_audio(audio)
    processed_length = len(processed_audio)

    print(f"原始音频长度: {original_length / sr:.2f}s, 处理后长度: {processed_length / sr:.2f}s")
    print(f"保留比例: {processed_length / original_length:.2f}")

    # 可视化原始与处理后的音频
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(audio)
    plt.title("原始音频")
    plt.subplot(2, 1, 2)
    plt.plot(processed_audio)
    plt.title("静音去除后音频")
    plt.tight_layout()
    plt.savefig("audio_processing.png")
    print("音频处理对比已保存为 audio_processing.png")


if __name__ == "__main__":
    # 依次运行测试
    test_feature_basic_properties()
    test_speaker_discrimination()
    test_cache_consistency()
    test_silence_removal()

    print("\n=== 测试完成 ===")
    print("请检查以下可能的问题点:")
    print("1. 特征是否包含NaN/无穷大值")
    print("2. 同一说话人特征相似度是否高于不同说话人")
    print("3. 缓存特征是否与原始特征一致")
    print("4. 静音去除是否保留了足够的语音信息")
