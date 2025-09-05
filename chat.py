import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def chat():
    """
    EmbeddingGemma を使った対話風のRAG検索システム (統計的手法版)
    """
    # --- 1. モデルのロード ---
    print("AI: モデルをロードしています... (初回は時間がかかることがあります)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer("google/embeddinggemma-300M").to(device=device)
    except Exception as e:
        print(f"AI: モデルのロード中にエラーが発生しました: {e}")
        return
    print("AI: モデルのロードが完了しました。")

    # --- 2. 検索対象のデータベース（ダミーの天気情報）を作成 ---
    documents = [
        "東京の天気は晴れです。最高気温は30度です。",
        "大阪の天気は雨で、一日中傘が必要です。",
        "横浜の天気は曇りのち晴れ。過ごしやすい一日でしょう。",
        "札幌は雪が降っており、気温は氷点下です。",
        "福岡は快晴で、絶好のお出かけ日和です。",
    ]

    # --- 3. データベースをベクトル化 ---
    print("AI: データベースを準備しています...")
    doc_embeddings = model.encode(documents)
    print("AI: 準備ができました。質問をどうぞ！ ('exit' または 'quit' で終了)")

    # --- 4. 対話ループ ---
    while True:
        try:
            # ユーザーからの質問を受け付ける
            query = input("You: ")

            if query.lower() in ["exit", "quit"]:
                print("AI: さようなら！")
                break

            if not query:
                continue

            # --- 5. 質問をベクトル化し、類似度検索を実行 ---
            query_embedding = model.encode(query)
            similarities = cos_sim(query_embedding, doc_embeddings)

            # --- 6. 結果の表示 (統計的手法版) ---
            scores = similarities[0]
            
            # numpyを使ってスコアの平均値を計算し、動的な閾値とする
            # .cpu() はGPU使用時に必要、.numpy()はnumpy配列に変換
            mean_score = np.mean(scores.cpu().numpy())

            # スコアとドキュメントをペアにしてリスト化
            scored_documents = []
            for i, score in enumerate(scores):
                scored_documents.append((documents[i], score.item())) # .item()でテンソルから数値に変換

            # スコアが高い順にソート
            scored_documents.sort(key=lambda x: x[1], reverse=True)

            print("AI (Debug): All document scores:")
            for doc, score in scored_documents:
                print(f"- '{doc}' (スコア: {score:.4f})")
            
            print(f"AI (Debug): Dynamic threshold (mean score) = {mean_score:.4f}")

            # 平均スコアを動的な閾値として使用
            print("\nAI: Based on the scores, I recommend:")
            found_something = False
            for doc, score in scored_documents:
                if score > mean_score:
                    print(f"- {doc}")
                    found_something = True
            
            if not found_something:
                print("- 申し訳ありませんが、関連する情報が見つかりませんでした。")

        except (KeyboardInterrupt, EOFError):
            print("\nAI: さようなら！")
            break
        except Exception as e:
            print(f"AI: エラーが発生しました: {e}")
            break


if __name__ == "__main__":
    chat()
