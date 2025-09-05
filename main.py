import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def main():
    """
    EmbeddingGemma を使った簡単な RAG (Retrieval-Augmented Generation) のサンプル
    """
    # --- 1. モデルのロード ---
    # 利用可能なデバイス（CUDAまたはCPU）を設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # EmbeddingGemma モデルをロード
    try:
        model = SentenceTransformer("google/embeddinggemma-300M").to(device=device)
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    # --- 2. [準備] 検索対象のデータベース（ダミーの天気情報）を作成 ---
    documents = [
        "東京の天気は晴れです。最高気温は30度です。",
        "大阪の天気は雨で、一日中傘が必要です。",
        "横浜の天気は曇りのち晴れ。過ごしやすい一日でしょう。",
        "札幌は雪が降っており、気温は氷点下です。",
        "福岡は快晴で、絶好のお出かけ日和です。",
    ]
    print("\n--- Search Database (Documents) ---")
    for doc in documents:
        print(f"- {doc}")

    # --- 3. [準備] データベースをベクトル化 ---
    # 各ドキュメントをベクトルに変換する
    print("\nVectorizing documents...")
    doc_embeddings = model.encode(documents)
    print("Vectorizing documents completed.")

    # --- 4. [実行] ユーザーからの質問 ---
    query = "横浜の天気は？"
    print(f"\n--- User Query ---\n{query}")

    # --- 5. [実行] 質問をベクトル化 ---
    query_embedding = model.encode(query)

    # --- 6. [実行] 類似度検索 ---
    # 質問ベクトルと、データベース内の全ベクトルとのコサイン類似度を計算
    similarities = cos_sim(query_embedding, doc_embeddings)

    # 最も類似度が高いドキュメントのインデックスを取得
    most_similar_idx = torch.argmax(similarities)
    highest_score = similarities[0][most_similar_idx]

    # --- 7. [実行] 結果の表示 ---
    print("\n--- Search Result ---")
    print(f"Most relevant document: '{documents[most_similar_idx]}'")
    print(f"Similarity score: {highest_score:.4f}")


if __name__ == "__main__":
    main()
