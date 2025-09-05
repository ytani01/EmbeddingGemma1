import torch
import click
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

@click.command()
@click.option(
    '--num-clusters',
    default=2,
    help='分類するグループの数。',
    show_default=True,
)
@click.argument(
    'input_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
def classify_command(num_clusters, input_dir):
    """
    指定されたフォルダ内のテキストファイルを読み込み、内容を意味に基づいて自動的に分類するCLIツール。
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

    # --- 4. クラスタリング ---
    if num_clusters > len(texts):
        print(f"\nエラー: グループ数({num_clusters})が文章数({len(texts)})より多くなっています。")
        return
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_assignment = clustering_model.fit_predict(embeddings)
    print(f"\nAI: {num_clusters}個のグループへの分類を実行しました。")

    # --- 5. 結果の表示 ---
    print("\n--- Classification Results ---")
    clustered_files = [[] for _ in range(num_clusters)]
    for i, cluster_id in enumerate(cluster_assignment):
        clustered_files[cluster_id].append(filenames[i])

    for i, cluster in enumerate(clustered_files):
        # グループ名をAI命名ではなく、クラスタID（0, 1, 2...）で表示する
        print(f"\n--- Cluster {i} ---")
        for filename in cluster:
            print(f"- {filename}")

if __name__ == "__main__":
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("エラー: scikit-learn がインストールされていません。")
        print("次のコマンドでインストールしてください: uv pip install scikit-learn")
        exit()
    
    classify_command()