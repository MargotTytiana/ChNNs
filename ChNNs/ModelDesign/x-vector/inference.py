import argparse
import torch
import numpy as np
from xvector_model import XVector
from FeatureExtractor import FeatureExtractor


def load_model(model_path, num_classes):
    model = XVector(input_dim=39, emb_dim=512, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def extract_features(audio_path, extractor):
    features = extractor.extract_features(audio_path)
    if features is not None:
        return torch.tensor(features).float().unsqueeze(0)  # 增加batch维度
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Speaker Identification Inference')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of speakers the model was trained on')
    args = parser.parse_args()

    # 初始化特征提取器
    extractor = FeatureExtractor(
        sample_rate=16000,
        n_mfcc=13,
        n_fft=512,
        hop_length=160,
        delta_order=2
    )

    # 加载模型
    model = load_model(args.model_path, args.num_classes)

    # 提取特征
    features = extract_features(args.audio_path, extractor)
    if features is None:
        print("Feature extraction failed.")
        return

    # 预测
    with torch.no_grad():
        outputs, embedding = model(features.transpose(1, 2))
        _, predicted = outputs.max(1)
        print(f"Predicted speaker ID: {predicted.item()}")
        print(f"Embedding: {embedding.squeeze().numpy()}")


if __name__ == '__main__':
    main()