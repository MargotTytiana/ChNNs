import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from data_loader import LibriSpeechDataset, extract_features
from chaos_features import ChaoticFeatureExtractor

# 配置参数 - 新增模型输入维度配置
TEST_CONFIG = {
    "sample_audio_dir": "P:/PycharmProjects/pythonProject1/dataset/Small_Libri",
    "segment_length": 1.0,
    "sampling_rate": 16000,
    "n_mfcc": 40,
    "embedding_dim": 2,
    "delay": 1,
    "mlsa_scales": [1, 2, 4],
    "n_lyapunov_exponents": 2,
    "rqa_threshold": None,
    "model_expected_input_dim": 120,  # 模型预期的输入特征维度
    "silence_removal_min_ratio": 0.3,  # 静音去除后最小保留比例阈值
    "normalization_tolerance": 1e-3  # 标准化容差范围
}


def test_feature_basic_properties():
    """增强特征基本属性测试，增加维度匹配检查"""
    print("=== 测试特征基本属性 ===")

    audio_files = librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])
    if not audio_files:
        print("Error: 未找到测试音频文件")
        return
    test_audio = audio_files[0]
    audio, sr = librosa.load(test_audio, sr=TEST_CONFIG["sampling_rate"])

    # 测试MFCC特征（含差分）
    mfcc_features = extract_features(audio, TEST_CONFIG["sampling_rate"])
    print(f"MFCC特征形状: {mfcc_features.shape}")  # (时间步, 120) 40*3
    print(f"MFCC特征总维度: {mfcc_features.shape[1]}")

    # 新增：特征维度与模型匹配检查
    if mfcc_features.shape[1] != TEST_CONFIG["model_expected_input_dim"]:
        print(f"警告: 特征维度与模型不匹配！特征维度={mfcc_features.shape[1]}, "
              f"模型预期={TEST_CONFIG['model_expected_input_dim']}")
    else:
        print("特征维度与模型匹配检查通过")

    # 原有异常值检查增强
    print(f"MFCC均值: {np.mean(mfcc_features):.4f}, 标准差: {np.std(mfcc_features):.4f}")
    print(f"MFCC含零比例: {np.mean(mfcc_features == 0):.4f}")
    print(f"MFCC含NaN: {np.isnan(mfcc_features).any()}, 含无穷大: {np.isinf(mfcc_features).any()}")

    # 混沌特征检查
    chaotic_extractor = ChaoticFeatureExtractor(
        embedding_dim=TEST_CONFIG["embedding_dim"],
        delay=TEST_CONFIG["delay"],
        scales=TEST_CONFIG["mlsa_scales"],
        n_lyapunov_exponents=TEST_CONFIG["n_lyapunov_exponents"],
        rqa_threshold=TEST_CONFIG["rqa_threshold"]
    )
    chaotic_features = chaotic_extractor.extract(audio)
    print(f"\n混沌特征形状: {chaotic_features.shape}")

    # 可视化MFCC特征
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_features.T[:40], sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("MFCC特征可视化")
    plt.tight_layout()
    plt.savefig("mfcc_visualization.png")
    print("MFCC可视化已保存为 mfcc_visualization.png")


def test_speaker_discrimination():
    """增强说话人区分能力测试，增加标准化前后对比"""
    print("\n=== 测试说话人区分能力 ===")

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
        n_lyapunov_exponents=TEST_CONFIG["n_lyapunov_exponents"],
        rqa_threshold=TEST_CONFIG["rqa_threshold"]  # 补充遗漏的参数
    )

    for speaker in speaker_dirs[:2]:
        spk_dir = os.path.join(TEST_CONFIG["sample_audio_dir"], speaker)
        audio_files = librosa.util.find_files(spk_dir, ext=["flac", "wav"])[:3]
        if not audio_files:
            continue

        mfcc_list = []
        chaotic_list = []
        for f in audio_files:
            audio, _ = librosa.load(f, sr=TEST_CONFIG["sampling_rate"])
            mfcc = extract_features(audio)
            chaotic = chaotic_extractor.extract(audio)
            mfcc_list.append(np.mean(mfcc, axis=0))
            chaotic_list.append(chaotic)

        # 新增：标准化特征（用于对比）
        scaler = StandardScaler()
        mfcc_norm = scaler.fit_transform(np.array(mfcc_list))

        speaker_features[speaker] = {
            "mfcc": np.array(mfcc_list),
            "mfcc_norm": mfcc_norm,  # 存储标准化后的特征
            "chaotic": np.array(chaotic_list)
        }
        print(f"说话人 {speaker} 提取了 {len(audio_files)} 个样本特征")

    # 计算相似度（增加标准化特征的对比）
    spk1, spk2 = speaker_dirs[:2]
    for feat_type in ["mfcc", "mfcc_norm", "chaotic"]:
        spk1_sim = np.mean(cosine_similarity(speaker_features[spk1][feat_type]))
        spk2_sim = np.mean(cosine_similarity(speaker_features[spk2][feat_type]))
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

    # t-SNE可视化（增加标准化特征）
    for feat_type in ["mfcc", "mfcc_norm"]:
        all_features = []
        all_labels = []
        for i, speaker in enumerate(speaker_dirs[:2]):
            all_features.extend(speaker_features[speaker][feat_type])
            all_labels.extend([i] * len(speaker_features[speaker][feat_type]))

        tsne = TSNE(n_components=2, perplexity=3)
        tsne_result = tsne.fit_transform(np.array(all_features))
        plt.figure(figsize=(8, 6))
        for label in set(all_labels):
            mask = np.array(all_labels) == label
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=speaker_dirs[label])
        plt.legend()
        plt.title(f"{feat_type}特征t-SNE分布")
        plt.savefig(f"{feat_type}_tsne.png")
        print(f"{feat_type}特征分布可视化已保存为 {feat_type}_tsne.png")


def test_cache_consistency():
    """保持缓存一致性测试"""
    print("\n=== 测试缓存一致性 ===")
    if not TEST_CONFIG["sample_audio_dir"]:
        return

    dataset = LibriSpeechDataset(
        audio_files=librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])[:2],
        root_dir=TEST_CONFIG["sample_audio_dir"],
        segment_length=TEST_CONFIG["segment_length"],
        sampling_rate=TEST_CONFIG["sampling_rate"],
        cache_dir="temp_cache"
    )

    item1 = dataset[0]
    item2 = dataset[0]

    # 比较两次加载的特征是否一致
    audio_diff = np.mean(np.abs(item1["audio"].numpy() - item2["audio"].numpy()))

    # 验证缓存一致性（差异应接近0）
    assert audio_diff < 1e-6, f"缓存特征不一致，平均差异: {audio_diff}"

    # 验证缓存文件是否正确创建
    assert os.path.exists(dataset._get_cache_path(dataset.audio_files[0])), "缓存文件未生成"

    print(f"缓存一致性验证通过，特征平均差异: {audio_diff}")


def test_silence_removal():
    """增强静音去除测试，增加多文件统计和阈值检查"""
    print("\n=== 测试静音去除 ===")
    audio_files = librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])[:5]  # 测试多个文件
    if not audio_files:
        return

    dataset = LibriSpeechDataset(
        audio_files=audio_files,
        root_dir=TEST_CONFIG["sample_audio_dir"],
        segment_length=TEST_CONFIG["segment_length"],
        sampling_rate=TEST_CONFIG["sampling_rate"]
    )

    # 修正：修复缩进错误，将处理逻辑放入函数内部
    savelist = []
    for f in audio_files:
        audio, sr = librosa.load(f, sr=TEST_CONFIG["sampling_rate"])
        original_length = len(audio)
        processed_audio = dataset._process_audio(audio)
        processed_length = len(processed_audio)
        ratio = processed_length / original_length
        savelist.append(ratio)
        print(f"文件 {os.path.basename(f)}: 原始长度 {original_length / sr:.2f}s, "
              f"处理后 {processed_length / sr:.2f}s, 保留比例 {ratio:.2f}")

    # 统计分析
    mean_ratio = np.mean(savelist)
    min_ratio = np.min(savelist)
    print(f"\n静音去除统计: 平均保留比例 {mean_ratio:.2f}, 最小保留比例 {min_ratio:.2f}")

    # 阈值检查
    if min_ratio < TEST_CONFIG["silence_removal_min_ratio"]:
        print(f"警告: 存在过度静音去除！最小保留比例 {min_ratio:.2f} < 阈值 {TEST_CONFIG['silence_removal_min_ratio']}")
    else:
        print("静音去除保留比例检查通过")

    # 可视化原始与处理后的音频（第一个文件）
    audio, sr = librosa.load(audio_files[0], sr=TEST_CONFIG["sampling_rate"])
    processed_audio = dataset._process_audio(audio)
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


def test_feature_normalization():
    """新增：特征标准化检查"""
    print("\n=== 测试特征标准化 ===")
    audio_files = librosa.util.find_files(TEST_CONFIG["sample_audio_dir"], ext=["flac", "wav"])[:5]
    if not audio_files:
        print("Error: 未找到测试音频文件")
        return

    # 收集多个样本的特征
    all_mfcc = []
    for f in audio_files:
        audio, sr = librosa.load(f, sr=TEST_CONFIG["sampling_rate"])
        mfcc = extract_features(audio)
        all_mfcc.append(mfcc)
    all_mfcc = np.concatenate(all_mfcc, axis=0)  # 合并所有时间步特征

    # 检查每个特征维度的均值和方差
    dim_means = np.mean(all_mfcc, axis=0)
    dim_stds = np.std(all_mfcc, axis=0)

    # 统计异常维度（均值偏离0或方差偏离1）
    bad_mean_dims = np.sum(np.abs(dim_means) > TEST_CONFIG["normalization_tolerance"])
    bad_std_dims = np.sum(np.abs(dim_stds - 1) > TEST_CONFIG["normalization_tolerance"])

    print(f"特征维度总数: {all_mfcc.shape[1]}")
    print(f"均值异常的维度数: {bad_mean_dims} (阈值 ±{TEST_CONFIG['normalization_tolerance']})")
    print(f"方差异常的维度数: {bad_std_dims} (阈值 1±{TEST_CONFIG['normalization_tolerance']})")

    # 可视化均值和方差分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(dim_means, bins=20)
    plt.title("各维度均值分布")
    plt.axvline(x=TEST_CONFIG["normalization_tolerance"], color='r', linestyle='--')
    plt.axvline(x=-TEST_CONFIG["normalization_tolerance"], color='r', linestyle='--')

    plt.subplot(1, 2, 2)
    plt.hist(dim_stds, bins=20)
    plt.title("各维度方差分布")
    plt.axvline(x=1 + TEST_CONFIG["normalization_tolerance"], color='r', linestyle='--')
    plt.axvline(x=1 - TEST_CONFIG["normalization_tolerance"], color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig("feature_normalization.png")
    print("标准化分布可视化已保存为 feature_normalization.png")

    if bad_mean_dims == 0 and bad_std_dims == 0:
        print("特征标准化检查通过")
    else:
        print("警告: 特征标准化存在偏差")


if __name__ == "__main__":
    test_feature_basic_properties()  # 包含维度匹配检查
    test_speaker_discrimination()  # 包含标准化前后对比
    test_silence_removal()  # 增强静音去除检查
    test_feature_normalization()  # 新增标准化专项测试
    test_cache_consistency()

    print("\n=== 测试完成 ===")
    print("请检查以下可能的问题点:")
    print("1. 特征是否包含NaN/无穷大值")
    print("2. 特征维度是否与模型输入层匹配")
    print("3. 同一说话人特征相似度是否高于不同说话人（尤其是标准化后）")
    print("4. 静音去除保留比例是否高于阈值（避免过度去除）")
    print("5. 特征各维度均值是否接近0、方差是否接近1")

