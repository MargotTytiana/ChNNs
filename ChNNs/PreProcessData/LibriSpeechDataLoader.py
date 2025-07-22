import os
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import logging
import psutil
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from audiomentations import Compose, AddBackgroundNoise, PitchShift
import torch
import json


def setup_logging(log_file='processing.log'):
    """配置日志系统"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# ======================
# 1. 数据格式统一化模块（支持多数据源）
# ======================
class AudioStandardizer:
    def __init__(self, target_sr=16000, output_format='wav', log_file='processing.log',
                 json_state_file='processed_files.json', save_interval=100):
        self.target_sr = target_sr
        self.output_format = output_format.lower()
        assert self.output_format in ('wav', 'flac'), "仅支持WAV/FLAC格式"
        self.start_time = time.time()
        self.last_time_report = self.start_time
        self.folder_count = 0
        self.log_file = log_file
        self.json_state_file = json_state_file
        self.save_interval = save_interval
        self.save_counter = 0
        self.processed_files = set()  # 存储"源标识|文件路径"格式的唯一标识

        # 加载已处理文件状态（兼容多数据源）
        if os.path.exists(self.json_state_file):
            try:
                with open(self.json_state_file, 'r', encoding='utf-8') as f:
                    self.processed_files = set(json.load(f))
                logging.info(f"从JSON状态文件恢复了 {len(self.processed_files)} 个已处理文件")
            except Exception as e:
                logging.warning(f"无法读取JSON状态文件: {e}")

    def save_progress_json(self):
        try:
            with open(self.json_state_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_files), f)
            logging.info(f"[状态保存] 已保存 {len(self.processed_files)} 个文件到JSON")
        except Exception as e:
            logging.error(f"[错误] 保存JSON状态失败: {str(e)}")

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def progress_report(self, files_processed, force_report=False):
        """进度报告函数"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        if force_report or (current_time - self.last_time_report > 3600):
            mem_usage = self.get_memory_usage()
            logging.info(
                f"[进度报告] 运行时间: {elapsed / 60:.2f}分钟 | "
                f"已处理文件夹: {self.folder_count} | "
                f"总文件数: {files_processed} | "
                f"内存使用: {mem_usage:.2f}MB"
            )
            self.last_time_report = current_time

        if self.folder_count % 50 == 0 and self.folder_count > 0:
            mem_usage = self.get_memory_usage()
            logging.info(
                f"[文件夹里程碑] 已处理 {self.folder_count} 个文件夹 | "
                f"当前文件数: {files_processed} | "
                f"内存使用: {mem_usage:.2f}MB"
            )

    def process_file(self, input_path, source_identifier, output_root):
        """处理单个音频文件（新增source_identifier区分数据源）"""
        # 生成唯一标识（数据源+文件路径），避免不同源文件重名冲突
        unique_id = f"{source_identifier}|{os.path.normpath(input_path)}"
        if unique_id in self.processed_files:
            logging.info(f"[跳过] 已在日志中处理: {input_path}")
            return None

        # 构建输出路径：在输出根目录下按数据源分类（避免路径冲突）
        rel_path = os.path.relpath(input_path, start=source_identifier)
        output_path = os.path.join(output_root, source_identifier.split(os.sep)[-1],
                                   rel_path.replace('.flac', f'.{self.output_format}'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 读取并统一格式
        try:
            y, _ = librosa.load(input_path, sr=self.target_sr, mono=True)
        except Exception as e:
            logging.error(f"[错误] 加载音频失败: {input_path} - {str(e)}")
            return None

        # 保存文件
        try:
            if self.output_format == 'wav':
                sf.write(output_path, y, self.target_sr, subtype='PCM_16')
            else:
                sf.write(output_path, y, self.target_sr)

            self.processed_files.add(unique_id)
            self.save_counter += 1
            if self.save_counter >= self.save_interval:
                self.save_progress_json()
                self.save_counter = 0
            return output_path

        except Exception as e:
            logging.error(f"[错误] 保存音频失败: {output_path} - {str(e)}")
            return None

    def process_multiple_corpus(self, corpus_roots, output_root):
        file_paths = []
        self.folder_count = 0
        files_processed = 0
        all_files = []

        for root in corpus_roots:
            root = os.path.normpath(root)
            if not os.path.exists(root):
                logging.warning(f"数据源路径不存在，已跳过：{root}")
                continue

            # 打印当前处理的数据源，确认dev路径被正确识别
            source_name = os.path.basename(root)  # 如"LibriSpeech"或"dev-clean"
            logging.info(f"开始收集数据源 {source_name} 的FLAC文件（路径：{root}）")

            # 遍历当前数据源的所有文件，强制收集FLAC
            flac_count = 0  # 统计当前数据源收集到的FLAC数量
            for subroot, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith('.flac'):  # 不区分大小写（避免FLAC/Flac等格式问题）
                        file_path = os.path.join(subroot, f)
                        all_files.append((file_path, root))
                        flac_count += 1

            # 关键：打印当前数据源收集到的FLAC数量（必须>0才正常）
            logging.info(f"数据源 {source_name} 共收集到 {flac_count} 个FLAC文件")
            if flac_count == 0:
                logging.error(f"数据源 {source_name} 未找到任何FLAC文件！请检查路径是否正确（是否包含FLAC文件）")

        # 处理所有收集到的文件（主数据集+dev）
        logging.info(f"总收集到 {len(all_files)} 个FLAC文件（含所有数据源）")
        with tqdm(total=len(all_files), desc="处理所有音频文件") as pbar:
            for input_path, source_identifier in all_files:
                output_path = self.process_file(input_path, source_identifier, output_root)
                if output_path:
                    file_paths.append(output_path)
                    files_processed += 1
                    # 打印dev文件的处理结果（用于验证）
                    if "dev-clean" in source_identifier:
                        logging.debug(f"已处理dev文件：{output_path}")  # 调试日志：确认dev文件被处理
                pbar.update(1)

        self.progress_report(files_processed, force_report=True)
        self.save_progress_json()
        logging.info(f"[完成] 共处理 {files_processed} 个文件（主数据集+dev）")
        return file_paths


# ======================
# 2. 元数据生成模块（支持多数据源合并）
# ======================
class MetadataGenerator:
    @staticmethod
    def parse_speakers_file(corpus_root):
        """解析SPEAKERS.TXT文件（兼容dev数据集格式 & 兼容不同数据集的文件位置）"""
        speakers_file = os.path.join(corpus_root, "SPEAKERS.TXT")

        # 检查文件是否存在，如果不存在则尝试dev数据集的位置
        if not os.path.exists(speakers_file):
            # 尝试dev数据集的SPEAKERS.TXT位置（假设在父目录）
            parent_dir = os.path.dirname(corpus_root)
            speakers_file = os.path.join(parent_dir, "SPEAKERS.TXT")

            if not os.path.exists(speakers_file):
                logging.warning(f"未找到SPEAKERS.TXT文件: {speakers_file}")
                return {}

        try:
            with open(speakers_file, encoding="utf-8") as f:
                speakers = {}
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(';'):  # 跳过注释和空行
                        continue

                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 5:
                        speaker_id = parts[0]
                        speakers[speaker_id] = {
                            'SEX': parts[1],
                            'SUBSET': parts[2],
                            'MINUTES': float(parts[3]) if parts[3].replace('.', '', 1).isdigit() else None,
                            'NAME': '|'.join(parts[4:])  # 支持带|的名字
                        }
            return speakers
        except Exception as e:
            logging.error(f"解析SPEAKERS.TXT失败: {str(e)}")
            return {}

    @staticmethod
    def parse_transcript(file_path):
        """解析.trans.txt转录文件"""
        try:
            with open(file_path, encoding='utf-8') as f:
                transcripts = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue

                    utt_id = parts[0].strip()
                    text = parts[1].strip()
                    transcripts[utt_id] = text
                return transcripts
        except Exception as e:
            logging.error(f"解析转录文件失败: {file_path} - {str(e)}")
            return {}

    @classmethod
    def build_metadata(cls, corpus_roots, audio_root, audio_format='wav'):
        all_metadata = []
        audio_root = os.path.normpath(audio_root)

        for root in corpus_roots:
            root = os.path.normpath(root)
            if not os.path.exists(root):
                logging.warning(f"数据源路径不存在，已跳过元数据生成：{root}")
                continue

            # 提取dev数据源名称（如"dev-clean"）
            source_name = os.path.basename(root)  # 关键：dev的source_name应为"dev-clean"
            logging.info(f"开始生成数据源 {source_name} 的元数据（原始路径：{root}）")

            # 解析SPEAKERS.TXT（dev的在父目录）
            speakers = cls.parse_speakers_file(os.path.dirname(root))  # dev的SPEAKERS在父目录
            logging.info(f"数据源 {source_name} 解析到 {len(speakers)} 个说话人信息")

            # 遍历dev的目录结构（子集→说话人→章节）
            dev_meta_count = 0  # 统计dev生成的元数据数量
            for subset_dir in os.listdir(root):
                subset_path = os.path.join(root, subset_dir)
                if not os.path.isdir(subset_path):
                    continue

                for speaker_id in os.listdir(subset_path):
                    speaker_path = os.path.join(subset_path, speaker_id)
                    if not os.path.isdir(speaker_path):
                        continue

                    for chapter_id in os.listdir(speaker_path):
                        chapter_path = os.path.join(speaker_path, chapter_id)
                        if not os.path.isdir(chapter_path):
                            continue

                        # 查找转录文件
                        trans_file = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
                        if not os.path.exists(trans_file):
                            continue
                        transcripts = cls.parse_transcript(trans_file)
                        if not transcripts:
                            continue

                        # 构建dev处理后的音频路径
                        for utt_id, text in transcripts.items():
                            # 处理后的路径：processed_data/dev-clean/子集/说话人/章节/utt_id.wav
                            audio_path = os.path.join(
                                audio_root,
                                source_name,  # 用"dev-clean"作为子目录
                                subset_dir,
                                speaker_id,
                                chapter_id,
                                f"{utt_id}.{audio_format}"
                            )

                            # 检查文件是否存在（并打印日志确认）
                            if not os.path.exists(audio_path):
                                logging.warning(f"[dev缺失] 处理后的音频不存在：{audio_path}（请检查音频处理是否成功）")
                                continue

                            # 读取时长（修复librosa参数警告）
                            try:
                                duration = librosa.get_duration(path=audio_path)  # 替换filename为path
                            except Exception as e:
                                logging.error(f"[dev错误] 读取时长失败：{audio_path} - {e}")
                                continue

                            # 添加dev的元数据条目
                            all_metadata.append({
                                'source': source_name,  # 明确标记为"dev-clean"
                                'speaker_id': speaker_id,
                                'gender': speakers.get(speaker_id, {}).get('SEX'),
                                'subset': 'dev-clean',  # 强制标记subset为dev-clean（确保后续能筛选）
                                'minutes': speakers.get(speaker_id, {}).get('MINUTES'),
                                'name': speakers.get(speaker_id, {}).get('NAME'),
                                'folder_subset': subset_dir,
                                'file_path': audio_path,
                                'duration': duration,
                                'transcript': text
                            })
                            dev_meta_count += 1

            # 打印dev生成的元数据数量（必须>0才正常）
            logging.info(f"数据源 {source_name} 生成 {dev_meta_count} 条元数据（若为0则无有效文件）")

        # 合并并打印总结果
        df = pd.DataFrame(all_metadata)
        logging.info(f"元数据生成：共合并 {len(df)} 条样本（主数据集+dev）")
        # 打印数据源分布（验证是否有dev-clean）
        if not df.empty:
            logging.info(f"数据源分布：\n{df['source'].value_counts()}")
        return df

    @staticmethod
    def analyze_metadata(df):
        """分析元数据统计信息（新增数据源分布）"""
        if df.empty:
            logging.warning("元数据为空，无法进行分析")
            return

        # 设置中文字体（修改部分）
        import matplotlib.font_manager as fm

        # 尝试查找系统中可用的中文字体
        chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
        font_path = None

        for font in chinese_fonts:
            try:
                font_path = fm.findfont(font, fallback_to_default=False)
                if font_path:
                    plt.rcParams['font.family'] = font
                    break
            except:
                continue

        # 如果没有找到中文字体，则使用默认字体，并添加警告
        if font_path is None:
            logging.warning("未找到中文字体，图表中的中文可能无法正确显示")
            plt.rcParams['font.family'] = ['sans-serif']
            # 添加字体回退选项
            plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'] + plt.rcParams[
                'font.sans-serif']

        # 确保负号正确显示
        plt.rcParams['axes.unicode_minus'] = False

        # 后续代码保持不变...
        print("\n=== 数据集统计 ===")
        print(f"总样本数: {len(df)}")
        print(f"说话人数: {df['speaker_id'].nunique()}")
        print(f"总音频时长: {df['duration'].sum() / 3600:.2f} 小时")
        print("\n数据源分布:")
        print(df['source'].value_counts())  # 新增：显示不同数据源的样本数

        # 音频时长统计
        print("\n音频时长分布:")
        print(df['duration'].describe())

        # 可视化
        plt.figure(figsize=(18, 12))

        # 1. 时长分布
        plt.subplot(221)
        df['duration'].hist(bins=50)
        plt.title("音频时长分布")
        plt.xlabel("秒数")
        plt.ylabel("数量")

        # 2. 说话人分布
        plt.subplot(222)
        speaker_counts = df['speaker_id'].value_counts()
        speaker_counts.hist(bins=30)
        plt.title("每位说话人的音频数量分布")
        plt.xlabel("音频数量")
        plt.ylabel("说话人数")

        # 3. 性别分布
        plt.subplot(223)
        gender_counts = df['gender'].value_counts()
        gender_counts.plot(kind='bar', color=['pink', 'lightblue'])
        plt.title("性别分布")
        plt.xlabel("性别")
        plt.ylabel("数量")

        # 4. 数据源分布（新增）
        plt.subplot(224)
        source_counts = df['source'].value_counts()
        source_counts.plot(kind='bar', color='lightgreen')
        plt.title("数据源分布")
        plt.xlabel("数据源")
        plt.ylabel("数量")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("metadata_analysis.png")
        plt.close()


# ======================
# 3. 数据增强模块
# ======================
class AudioAugmenter:
    def __init__(self, sample_rate=16000, background_paths=None):
        self.sample_rate = sample_rate

        # 背景噪声路径（默认值仅为示例，需替换为实际路径）
        if background_paths is None:
            background_paths = ["backgroundNoise/musan/noise/free-sound"]
            logging.warning("使用默认背景噪声路径，建议替换为实际路径")

        self.transform = Compose([
            AddBackgroundNoise(
                sounds_path=background_paths,
                noise_rms="relative",
                min_snr_db=3.0,
                max_snr_db=30.0,
                p=0.5
            ),
            PitchShift(
                min_semitones=-2,
                max_semitones=2,
                p=0.3,
            ),
        ])

    def __call__(self, audio: np.ndarray):
        # 转换为numpy数组处理，输出转回Tensor
        samples = audio if isinstance(audio, np.ndarray) else audio.numpy()
        augmented = self.transform(samples=samples, sample_rate=self.sample_rate)
        return torch.from_numpy(augmented)


# ======================
# 4. 数据集类 (兼容多源数据)
# ======================
class LibriSpeechDataset(Dataset):
    def __init__(self, metadata, subset=None, chunk_size=48000,
                 augment=False, background_paths=None):
        """
        Args:
            metadata: 合并后的元数据DataFrame
            subset: 可选，指定子集（如'train-clean-100'）；None表示使用所有
            chunk_size: 裁剪长度(样本数)
            augment: 是否启用数据增强
            background_paths: 背景噪声路径列表
        """
        # 筛选子集（支持多源数据）
        self.metadata = metadata.copy()
        if subset is not None:
            self.metadata = self.metadata[self.metadata['subset'] == subset]

        self.chunk_size = chunk_size
        self.augment = augment
        self.augmenter = AudioAugmenter(background_paths=background_paths) if augment else None

        # 构建说话人到标签的映射（含所有数据源）
        self.speaker_to_idx = {sp: i for i, sp in enumerate(self.metadata['speaker_id'].unique())}
        self.metadata['label'] = self.metadata['speaker_id'].map(self.speaker_to_idx)

        logging.info(f"数据集初始化：{subset if subset else '所有数据'} 子集共 {len(self)} 个样本")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        try:
            audio, _ = librosa.load(row['file_path'], sr=16000, mono=True)
        except Exception as e:
            logging.error(f"加载音频失败: {row['file_path']} - {str(e)}")
            audio = np.zeros(self.chunk_size if self.chunk_size else 16000)  # 静音替代

        # 随机裁剪或补零
        if self.chunk_size is not None:
            if len(audio) >= self.chunk_size:
                start = np.random.randint(0, len(audio) - self.chunk_size)
                audio = audio[start:start + self.chunk_size]
            else:
                audio = np.pad(audio, (0, self.chunk_size - len(audio)), 'constant')

        # 数据增强
        if self.augment and self.augmenter:
            audio = self.augmenter(audio)

        return {
            'audio': torch.FloatTensor(audio),
            'label': row['label'],
            'speaker_id': row['speaker_id'],
            'duration': row['duration'],
            'source': row['source']  # 新增：返回数据源标识
        }

    def add_new_speakers(self, new_metadata):
        """持续学习：添加新说话人（支持多源新增数据）"""
        old_speakers = set(self.speaker_to_idx.keys())
        new_speakers = set(new_metadata['speaker_id'].unique()) - old_speakers

        # 更新标签映射
        max_idx = max(self.speaker_to_idx.values()) if self.speaker_to_idx else -1
        for i, sp in enumerate(new_speakers, start=max_idx + 1):
            self.speaker_to_idx[sp] = i

        # 合并元数据
        new_metadata = new_metadata.copy()
        new_metadata['label'] = new_metadata['speaker_id'].map(self.speaker_to_idx)
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)
        logging.info(f"新增说话人后：总类别数 {len(self.speaker_to_idx)}，总样本数 {len(self)}")


# ======================
# 5. 使用示例（多数据源处理）
# ======================
if __name__ == "__main__":
    # 配置路径（核心修改：定义多个数据源）
    CORPUS_ROOTS = [
        r"P:\PycharmProjects\pythonProject1\dataset\LibriSpeech",  # 主数据集
        r"P:\PycharmProjects\pythonProject1\devDataset\LibriSpeech\dev-clean"  # dev数据集
    ]
    OUTPUT_ROOT = r"P:\PycharmProjects\pythonProject1\processed_data"  # 统一输出目录
    BACKGROUND_PATHS = [
        r"P:\PycharmProjects\pythonProject1\backgroundNoise\musan\noise\free-sound",
        r"P:\PycharmProjects\pythonProject1\backgroundNoise\musan\noise\sound-bible"
    ]

    # 确保输出目录存在
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 初始化日志
    setup_logging(os.path.join(OUTPUT_ROOT, "processing.log"))
    logging.info("=== 开始处理多数据源LibriSpeech数据集 ===")

    try:
        # 1. 数据标准化（处理所有数据源）
        logging.info("开始统一所有数据源的音频格式...")
        standardizer = AudioStandardizer(output_format='wav')
        processed_files = standardizer.process_multiple_corpus(CORPUS_ROOTS, OUTPUT_ROOT)
        logging.info(f"成功处理 {len(processed_files)} 个音频文件（含主数据集和dev数据集）")

        # 2. 生成合并元数据
        logging.info("开始生成合并元数据...")
        metadata = MetadataGenerator.build_metadata(
            corpus_roots=CORPUS_ROOTS,
            audio_root=OUTPUT_ROOT,
            audio_format='wav'
        )

        # 保存元数据
        metadata_path = os.path.join(OUTPUT_ROOT, "metadata_all.csv")
        metadata.to_csv(metadata_path, index=False)
        logging.info(f"合并元数据已保存到 {metadata_path}")

        # 3. 分析元数据（含数据源分布）
        logging.info("开始分析元数据统计信息...")
        MetadataGenerator.analyze_metadata(metadata)

        # 4. 创建数据集（示例：train和dev）
        logging.info("创建训练数据集...")
        train_set = LibriSpeechDataset(
            metadata,
            subset='train-clean-100',  # 主数据集的训练子集
            augment=True,
            background_paths=BACKGROUND_PATHS
        )

        logging.info("创建dev数据集...")
        dev_set = LibriSpeechDataset(
            metadata,
            subset='dev-clean',  # dev数据集的子集
            augment=False  # 验证集不增强
        )

        # 示例：查看样本
        if len(train_set) > 0:
            sample = train_set[0]
            logging.info(f"\n训练集示例: 说话人 {sample['speaker_id']}，数据源 {sample['source']}")

        if len(dev_set) > 0:
            sample = dev_set[0]
            logging.info(f"Dev集示例: 说话人 {sample['speaker_id']}，数据源 {sample['source']}")

        # 5. 持续学习示例（添加dev数据到训练集）
        logging.info("\n模拟持续学习场景：将dev数据添加到训练集...")
        dev_metadata = metadata[metadata['subset'] == 'dev-clean']
        train_set.add_new_speakers(dev_metadata)

        logging.info("=== 所有处理完成 ===")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise