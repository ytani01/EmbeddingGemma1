import torch
import click
import os
from sentence_transformers import SentenceTransformer
import numpy as np

@click.command()
@click.option(
    '--min-cluster-size',
    default=2,
    help='A cluster must have at least this many samples.',
    show_default=True,
)
@click.option(
    '--min-samples',
    default=1,
    help='The number of samples in a neighborhood for a point to be considered as a core point.',
    show_default=True,
)
@click.option(
    '--force-no-noise',
    is_flag=True,
    help='Force all points into clusters, leaving no noise points.',
)
@click.argument(
    'input_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
def classify_command(min_cluster_size, min_samples, force_no_noise, input_dir):
    """
    指定されたフォルダ内のテキストファイルを読み込み、HDBSCANで内容を意味に基づいて自動的に分類するCLIツール。
    """
    # --- 1. ファイル読み込み ---
    texts, filenames = [], []
    print(f"AI: '{input_dir}' からテキストファイルを読み込んでいます...")
    try:
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    filenames.append(filename)
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました: {e}")
        return
    if not texts:
        print("エラー: 指定されたディレクトリに .txt ファイルが見つかりません。")
        return

    # --- 2. モデルロード ---
    print("AI: EmbeddingGemmaモデルをロードしています...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-300M").to(device=device)
    print("AI: モデルのロードが完了しました。")

    # --- 3. ベクトル化 ---
    print("\nAI: 文章をベクトルに変換しています...")
    embeddings = model.encode(texts)

    # --- 4. クラスタリング (HDBSCAN) ---
    print("\nAI: HDBSCANでクラスタリングを実行しています...")
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        print("エラー: hdbscan ライブラリが見つかりません。")
        print("プロジェクトの依存関係をインストールするために、'uv sync' を実行してください。")
        return

    # min_cluster_size, min_samples を調整することで、クラスタの挙動を制御できます
    print(f"AI: HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}) でクラスタリングを実行しています...")
    clustering_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    cluster_assignment = clustering_model.fit_predict(embeddings)

    # --force-no-noise オプションが指定された場合、ノイズを強制的にクラスタ化する
    if force_no_noise and -1 in cluster_assignment:
        print("AI: ノイズポイントを強制的にクラスタに割り当てています...")
        next_cluster_id = np.max(cluster_assignment) + 1
        noise_indices = np.where(cluster_assignment == -1)[0]
        for i in range(len(noise_indices)):
            cluster_assignment[noise_indices[i]] = next_cluster_id
            next_cluster_id += 1
    
    num_clusters = len(np.unique(cluster_assignment)) - (1 if -1 in cluster_assignment else 0)
    print(f"\nAI: {num_clusters}個のグループへの分類が完了しました。")


    # --- 5. 結果の表示 ---
    print("\n--- Classification Results ---")
    
    # クラスタIDごとにファイル名を整理
    clustered_files = {}
    for i, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_files:
            clustered_files[cluster_id] = []
        clustered_files[cluster_id].append(filenames[i])

    # クラスタIDでソートして表示
    for cluster_id in sorted(clustered_files.keys()):
        if cluster_id == -1:
            print("\n--- Noise (Unclassified) ---")
        else:
            print(f"\n--- Cluster {cluster_id} ---")
        
        for filename in clustered_files[cluster_id]:
            print(f"- {filename}")

if __name__ == "__main__":
    try:
        # HDBSCANの依存関係をチェック
        from hdbscan import HDBSCAN
    except ImportError:
        print("エラー: hdbscan ライブラリが見つかりません。")
        print("プロジェクトの依存関係をインストールするために、'uv sync' を実行してください。")
        exit()
    
    classify_command()
