# EmbeddingGemma サンプルプログラム集

このリポジトリは、Googleの軽量な埋め込みモデル「EmbeddingGemma」の様々な使い方を示すサンプルプログラム集です。

## == ✨ 概要

EmbeddingGemmaは、テキストを「ベクトル」と呼ばれる数値の配列に変換することに特化したモデルです。このベクトルを使うことで、コンピュータは文章の意味の近さを計算できるようになります。このリポジリでは、その能力を応用した具体的なサンプルを体験できます。

### === 収録されているサンプル

1.  **`chat.py` - 対話風セマンティック検索**
    *   あらかじめ用意された情報の中から、ユーザーの質問文と意味が最も近いものを探し出して回答する、対話風の検索ボットです。
    *   固定の閾値ではなく、スコアの平均値を使った動的な絞り込みロジックを実装しています。

2.  **`classify_texts.py` - 文章の自動分類（K-Means）**
    *   「食べ物」「スポーツ」といった異なるテーマの文章が混在したリストを、AIが意味の近さに基づいて自動的に指定された数のグループに分類します。

3.  **`classify_texts_hdbscan.py` - 文章の自動分類（HDBSCAN）**
    *   密度ベースの手法であるHDBSCANを使用し、クラスタ数を自動で決定して文章を分類します。パラメータ調整により、ノイズの扱いを柔軟に制御できます。

## == 🚀 セットアップ方法

このプロジェクトは、Pythonの高速なパッケージ管理ツール `uv` を使用することを前提としています。

### === 1. リポジトリのクローン

```bash
git clone https://github.com/ytani01/EnbeddingGemma1
cd embeddinggemma1
```

### === 2. 仮想環境の作成と依存関係のインストール

`uv`は、プロジェクト内に仮想環境 `.venv` がない場合、以下のコマンド実行時に自動で作成してくれます。

`uv sync` コマンドを実行すると、仮想環境の準備と、`pyproject.toml` に基づくライブラリのインストールが一度に行われます。

```bash
# このコマンド一発で仮想環境の作成とライブラリのインストールが完了します
uv sync
```

もし、仮想環境の作成を明示的に行いたい場合は、先に `uv venv` を実行することも可能です。

```bash
# 1. 仮想環境を明示的に作成
uv venv

# 2. ライブラリをインストール
uv sync
```

### === 3. Hugging Face Hub へのログイン

EmbeddingGemmaモデルは、Hugging Face Hub上でアクセスが制限されています。利用するには、Hugging Faceのアカウントでログインし、モデルへのアクセス権を取得する必要があります。

**a. モデルへのアクセスリクエスト:**

まず、ウェブブラウザで以下のページにアクセスし、利用規約に同意してアクセスをリクエストしてください。

*   [google/embeddinggemma-300M](https://huggingface.co/google/embeddinggemma-300M)

**b. トークンの準備とログイン:**

次に、Hugging Faceのアクセストークンを準備します。
[こちらのページ](https://huggingface.co/settings/tokens) から、`read` 権限を持つトークンをコピーしてください。

準備ができたら、ターミナルで以下のコマンドを実行してログインします。（この `hf` コマンドは、`huggingface-hub` パッケージに含まれており、`uv sync` によってインストール済みです）

```bash
uv run hf auth login
```

`Token:` と表示されたら、コピーしたトークンを貼り付けてEnterキーを押してください。

## == 使い方

セットアップが完了したら、各サンプルを試すことができます。

### === 対話風検索ボットの実行 (`chat.py`)

```bash
uv run python chat.py
```

`You:` と表示されたら、「晴れなのはどこ？」「天気が悪いのはどこ？」などの質問を入力してみてください。
終了するには `exit` または `quit` と入力します。

### === 文章の自動分類の実行 (`classify_texts.py`)

`click`ライブラリを導入したことにより、コマンドラインから操作できます。
分類したいテキストファイル（`.txt`）を一つのフォルダにまとめ、そのフォルダのパスを指定して実行します。

**基本的な使い方:**

```bash
uv run python classify_texts.py <フォルダのパス>
```

**実行例:**

リポジトリ内にサンプルデータとして `text_classification_data` フォルダを用意しています。
以下のコマンドで、このフォルダ内のテキストファイルを2つのグループに分類できます。

```bash
uv run python classify_texts.py text_classification_data
```

**グループ数を指定する場合:**

`--num-clusters` オプションで、分類するグループの数を変更できます。

```bash
# 3つのグループに分類する例
uv run python classify_texts.py --num-clusters 3 text_classification_data
```

### === HDBSCANによる文章の自動分類 (`classify_texts_hdbscan.py`)

このスクリプトは、クラスタ数を自動決定するHDBSCANアルゴリズムを使用します。
`pyproject.toml` に依存関係が定義されているため、セットアップ時の `uv sync` コマンドで必要な `hdbscan` ライブラリもインストールされます。

**基本的な使い方:**

```bash
uv run python classify_texts_hdbscan.py text_classification_data/
```

**パラメータチューニング:**

HDBSCANの挙動を細かく制御するためのオプションが用意されています。

*   `--min-cluster-size <数>`: クラスタを構成する最小のサンプル数を指定します。（デフォルト: 2）
*   `--min-samples <数>`: 点がクラスタの核（コア）と見なされるために必要な近傍のサンプル数を指定します。この値を大きくすると、より密な領域のみがクラスタ化され、ノイズが増える傾向があります。（デフォルト: 1）
*   `--force-no-noise`: このフラグを付けると、ノイズと判定された全ての点を、それぞれ独立した単一のクラスタとして強制的に分類します。

**実行例1: クラスタ形成の条件を厳しくする**

（ノイズが増える可能性があります）

```bash
uv run python classify_texts_hdbscan.py --min-samples 2 text_classification_data/
```

**実行例2: 全てのファイルをいずれかのクラスタに強制的に分類する**

```bash
uv run python classify_texts_hdbscan.py --force-no-noise text_classification_data/
```

**ヘルプの表示:**

```bash
uv run python classify_texts_hdbscan.py --help
```
