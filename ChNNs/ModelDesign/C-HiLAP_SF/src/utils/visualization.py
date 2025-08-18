import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Union, Optional
import plotly.graph_objects as go

# 设置绘图风格
sns.set_style("whitegrid")


def plot_loss_curves(
        history: Dict[str, List[float]],
        title: str = "Training and Validation Loss",
        figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    绘制训练和验证损失曲线。

    Args:
        history: 包含损失历史的字典，例如 {'train_loss': [...], 'val_loss': [...]}
        title: 图表标题
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')

    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_det_curve(
        scores_genuine: np.ndarray,
        scores_impostor: np.ndarray,
        title: str = "Detection Error Tradeoff (DET) Curve",
        figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    绘制检测错误权衡(DET)曲线。

    Args:
        scores_genuine: 合法配对的相似度分数
        scores_impostor: 冒充配对的相似度分数
        title: 图表标题
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    from metrics import compute_eer  # 导入EER计算函数

    # 确保输入是numpy数组
    scores_genuine = np.asarray(scores_genuine).flatten()
    scores_impostor = np.asarray(scores_impostor).flatten()

    # 创建标签数组
    labels = np.concatenate([np.ones_like(scores_genuine), np.zeros_like(scores_impostor)])
    scores = np.concatenate([scores_genuine, scores_impostor])

    # 计算不同阈值下的FAR和FRR
    thresholds = np.sort(np.unique(scores))
    fars, frrs = [], []

    for threshold in thresholds:
        pred = scores >= threshold
        far = np.sum((pred == 1) & (labels == 0)) / len(scores_impostor)
        frr = np.sum((pred == 0) & (labels == 1)) / len(scores_genuine)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制DET曲线
    ax.plot(fars, frrs, 'b-', lw=2, label='DET Curve')

    # 绘制EER点
    eer, _ = compute_eer(scores_genuine, scores_impostor)
    ax.plot([eer], [eer], 'ro', markersize=8, label=f'EER = {eer:.2%}')

    # 设置坐标轴
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('False Rejection Rate (FRR)')
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlim([0.001, 1.0])
    ax.set_ylim([0.001, 1.0])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_embedding_space(
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization of Speaker Embeddings",
        method: str = 'tsne',
        n_components: int = 2,
        figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    使用t-SNE或PCA可视化高维嵌入空间。

    Args:
        embeddings: 嵌入向量，形状为 (n_samples, n_features)
        labels: 标签，形状为 (n_samples,)
        title: 图表标题
        method: 降维方法 ('tsne' 或 'pca')
        n_components: 目标维度 (2或3)
        figsize: 图表大小

    Returns:
        matplotlib图表对象
    """
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, perplexity=30.0, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # 创建图表
    fig = plt.figure(figsize=figsize)

    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")

        # 添加颜色条
        legend = ax.legend(*scatter.legend_elements(), title="Speakers")
        ax.add_artist(legend)

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")
        ax.set_zlabel(f"{method.upper()} Dimension 3")

        # 添加颜色条
        legend = ax.legend(*scatter.legend_elements(), title="Speakers")
        ax.add_artist(legend)

    else:
        raise ValueError("n_components must be 2 or 3")

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_phase_space_3d_interactive(
        trajectory: np.ndarray,
        title: str = "Interactive 3D Phase Space Trajectory"
) -> go.Figure:
    """
    使用Plotly绘制交互式的3D相空间轨迹。

    Args:
        trajectory: 相空间轨迹，形状为 (n_points, 3)
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='lines',
        line=dict(
            color='blue',
            width=2
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title='x(t+τ)',
            zaxis_title='x(t+2τ)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def plot_recurrence_plot_interactive(
        rp_matrix: np.ndarray,
        title: str = "Interactive Recurrence Plot"
) -> go.Figure:
    """
    使用Plotly绘制交互式的递归图。

    Args:
        rp_matrix: 递归图矩阵
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    fig = go.Figure(data=go.Heatmap(
        z=rp_matrix,
        colorscale='gray',
        showscale=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Index",
        yaxis_title="Time Index",
        yaxis_autorange='reversed'
    )

    return fig


def plot_bifurcation_diagram(
        map_function,
        param_name: str,
        param_range: Tuple[float, float],
        num_param_steps: int = 1000,
        num_iterations: int = 1000,
        num_plot_points: int = 200,
        title: str = "Bifurcation Diagram"
) -> plt.Figure:
    """
    绘制混沌映射的分岔图。

    Args:
        map_function: 混沌映射函数，形式为 f(x, param)
        param_name: 参数名称
        param_range: 参数范围
        num_param_steps: 参数步数
        num_iterations: 迭代次数
        num_plot_points: 绘制的点数
        title: 图表标题

    Returns:
        matplotlib图表对象
    """
    params = np.linspace(param_range[0], param_range[1], num_param_steps)
    x = 0.1 * np.ones(num_param_steps)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 迭代以达到稳定状态
    for _ in range(num_iterations):
        x = map_function(x, params)

    # 绘制分岔点
    for _ in range(num_plot_points):
        x = map_function(x, params)
        ax.plot(params, x, ',k', alpha=0.25)

    ax.set_xlim(param_range)
    ax.set_title(title)
    ax.set_xlabel(f"Parameter '{param_name}'")
    ax.set_ylabel("x")

    plt.tight_layout()
    return fig


def plot_attractor_comparison_interactive(
        attractors: Dict[str, np.ndarray],
        title: str = "Interactive Comparison of Speaker Attractors"
) -> go.Figure:
    """
    使用Plotly交互式地比较不同说话人的吸引子。

    Args:
        attractors: 字典，键为说话人ID，值为吸引子轨迹
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    for speaker_id, trajectory in attractors.items():
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines',
            name=f"Speaker {speaker_id}",
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        legend_title="Speakers",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


# 示例用法
if __name__ == "__main__":
    # --- 示例1: 绘制损失曲线 ---
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'val_loss': [0.6, 0.5, 0.45, 0.4, 0.38]
    }
    fig_loss = plot_loss_curves(history)
    fig_loss.savefig("loss_curves.png")
    print("损失曲线已保存到 loss_curves.png")

    # --- 示例2: 绘制DET曲线 ---
    scores_genuine = np.random.normal(0.7, 0.2, 1000)
    scores_impostor = np.random.normal(0.3, 0.2, 1000)
    fig_det = plot_det_curve(scores_genuine, scores_impostor)
    fig_det.savefig("det_curve_visualization.png")
    print("DET曲线已保存到 det_curve_visualization.png")

    # --- 示例3: 绘制嵌入空间 ---
    n_speakers = 5
    n_samples_per_speaker = 50
    embeddings = np.random.rand(n_speakers * n_samples_per_speaker, 192)
    labels = np.repeat(np.arange(n_speakers), n_samples_per_speaker)
    fig_tsne = plot_embedding_space(embeddings, labels)
    fig_tsne.savefig("embedding_space.png")
    print("嵌入空间可视化已保存到 embedding_space.png")

    # --- 示例4: 绘制交互式3D相空间轨迹 ---
    t = np.linspace(0, 50, 5000)
    x = np.sin(t)
    y = np.cos(t)
    z = np.sin(2 * t)
    trajectory = np.vstack([x, y, z]).T
    fig_phase_space = plot_phase_space_3d_interactive(trajectory)
    fig_phase_space.write_html("phase_space_interactive.html")
    print("交互式3D相空间轨迹已保存到 phase_space_interactive.html")


    # --- 示例5: 绘制分岔图 ---
    def logistic_map(x, r):
        return r * x * (1 - x)


    fig_bifurcation = plot_bifurcation_diagram(
        logistic_map,
        param_name='r',
        param_range=(2.5, 4.0),
        title="Bifurcation Diagram of Logistic Map"
    )
    fig_bifurcation.savefig("bifurcation_diagram.png")
    print("分岔图已保存到 bifurcation_diagram.png")

    # --- 示例6: 比较吸引子 ---
    attractors = {
        'A': np.random.rand(100, 3),
        'B': np.random.rand(100, 3) + 1.0
    }
    fig_attractor_comp = plot_attractor_comparison_interactive(attractors)
    fig_attractor_comp.write_html("attractor_comparison.html")
    print("吸引子比较图已保存到 attractor_comparison.html")