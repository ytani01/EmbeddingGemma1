import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def classify_texts():
    """
    EmbeddingGemma を使って文章を意味に基づいて自動的に分類（クラスタリング）するサンプル
    """
    # --- 1. モデルのロード ---
    # 文章をベクトルに変換するためのEmbeddingGemmaモデルをロードする。
    # GPUが利用可能ならGPUを、そうでなければCPUを使用する。
    print("AI: モデルをロードしています...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer("google/embeddinggemma-300M").to(device=device)
    except Exception as e:
        print(f"AI: モデルのロード中にエラーが発生しました: {e}")
        return
    print("AI: モデルのロードが完了しました。")

    # --- 2. 分類対象の文章 ---
    # ここでは例として、「食べ物」と「スポーツ」に関する文章が混在したリストを定義する。
    # これらの文章が、AIによって意味的に近いグループに分けられるかを確認する。
    texts = [
        "昨日食べたラーメンはとても美味しかった。",
        "サッカーの試合は後半に劇的なゴールが決まった。",
        "新鮮な魚介を使った寿司は格別だ。",
        "彼は毎朝公園でジョギングをしている。",
        "デザートに濃厚なチョコレートケーキを注文した。",
        "テニスの大会で彼は見事優勝を果たした。",
    ]

    print("\n--- Input Texts ---")
    for text in texts:
        print(f"- {text}")

    # --- 3. 文章をベクトル化 ---
    # 用意した文章リストをEmbeddingGemmaモデルに渡し、それぞれの文章を
    # 意味を表現する数値のベクトル（768次元）に変換する。
    # このベクトル間の距離が、文章間の意味の近さを表す。
    print("\nAI: 文章をベクトルに変換しています...")
    embeddings = model.encode(texts)
    print(f"AI: ベクトル化が完了しました。 (Shape: {embeddings.shape})")

    # --- 4. K-meansクラスタリングで分類 ---
    # scikit-learnライブラリのKMeansアルゴリズムを使い、ベクトルをグループ分けする。
    # n_clustersで、いくつのグループに分けるかを指定する。
    # random_stateは、毎回同じ結果を得るための乱数シード。
    # n_initは、異なる初期値でアルゴリズムを10回実行し、最も良い結果を採用する設定。
    num_clusters = 2
    clustering_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    # .fit()で、ベクトルデータを使ってクラスタリングモデルを学習させる。
    clustering_model.fit(embeddings)
    # .labels_で、各文章がどのクラスタID（0か1）に分類されたかの結果を取得する。
    cluster_assignment = clustering_model.labels_

    print(f"\nAI: {num_clusters}個のグループへの分類を実行しました。")

    # --- 5. 結果の表示 ---
    # 分類結果を人間が分かりやすいように表示する。
    # まず、分類先のグループの数だけ空のリストを作成する。
    clustered_sentences = [[] for i in range(num_clusters)]
    # 各文章の分類結果（cluster_id）を見て、対応するグループのリストに文章を追加していく。
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(texts[sentence_id])

    # 最終的なグループ分けの結果を表示する。
    print("\n--- Classification Results ---")
    for i, cluster in enumerate(clustered_sentences):
        print(f"\n--- Group {i+1} ---")
        for sentence in cluster:
            print(f"- {sentence}")

if __name__ == "__main__":
    # このサンプルを実行するには、scikit-learnライブラリが必要です。
    # プログラムがライブラリをインポートできるか試してみて、
    # もし失敗（ImportError）したら、インストール方法を案内して終了する。
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("エラー: scikit-learn がインストールされていません。")
        print("次のコマンドでインストールしてください: uv pip install scikit-learn")
        exit()
    
    classify_texts()