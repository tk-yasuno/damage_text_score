"""
クイックスタート実行スクリプト
サンプル画像でパイプラインを試す
"""
import os
import sys
import argparse
import io
from pathlib import Path

# llama.cppのログを抑制（文字化け防止）
os.environ['LLAMA_CPP_LOG_LEVEL'] = '0'

# Windows PowerShellのcp932エンコーディング問題を回避
if sys.platform == 'win32':
    # コンソール出力をUTF-8に設定
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # PowerShellコンソールをUTF-8に設定
    os.system('chcp 65001 > nul 2>&1')  # UTF-8コードページに変更

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from pipeline.end_to_end import DamageAnalysisPipeline


def main():
    """クイックスタート実行"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='損傷分析パイプライン クイックスタート')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3, 4],
                       help='処理モード: 1=単一画像(1枚), 2=小規模(10枚), 3=中規模(50枚), 4=全画像(254枚)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("損傷読解・補修優先度スコアリングシステム - クイックスタート")
    print("=" * 70)
    print()
    
    # パイプライン初期化
    config_path = project_root / 'config.yaml'
    pipeline = DamageAnalysisPipeline(str(config_path))
    
    # 入力データ
    data_dir = project_root / 'data' / 'images_human_inspect_n254'
    output_dir = project_root / 'data' / 'outputs'
    
    # データ確認
    image_files = list(data_dir.glob('*.png'))
    print(f"\n📁 データディレクトリ: {data_dir}")
    print(f"📊 画像数: {len(image_files)}枚")
    
    # ユーザー選択
    if args.mode:
        choice = str(args.mode)
        print(f"\n選択されたモード: {choice}")
    else:
        print("\n処理モードを選択してください：")
        print("  1. 単一画像テスト（1枚）")
        print("  2. 小規模テスト（10枚）")
        print("  3. 中規模テスト（50枚）")
        print("  4. 全画像処理（254枚）")
        print()
        choice = input("選択 [1-4]: ").strip()
    
    try:
        if choice == '1':
            # 単一画像
            test_img = image_files[0]
            print(f"\n🔍 テスト画像: {test_img.name}")
            
            result = pipeline.process_image(test_img)
            
            if result.status == "success":
                print(f"\n✅ 処理成功！")
                print(f"⏱️  処理時間: {result.processing_time:.2f}秒")
                print(f"\n📝 損傷説明:")
                print(result.description[:300] + '...')
                print(f"\n📊 構造化データ:")
                print(f"  損傷種別: {result.structure['damage_type']}")
                print(f"  重症度: {result.structure['severity']}")
                print(f"  位置: {result.structure['location']}")
                print(f"  リスク: {result.structure['risk']}")
                print(f"\n🎯 スコアリング結果:")
                print(f"  生スコア: {result.score['raw_score']:.3f}")
                print(f"  優先度: {result.score['priority_level']}")
                print(f"  説明: {result.score['priority_description']}")
            else:
                print(f"\n❌ エラー: {result.error}")
            
            # 結果保存
            output_dir.mkdir(parents=True, exist_ok=True)
            pipeline.save_results([result], output_dir, 'quickstart_single.csv')
        
        elif choice in ['2', '3', '4']:
            # 一括処理
            limits = {'2': 10, '3': 50, '4': None}
            limit = limits[choice]
            
            print(f"\n🚀 一括処理開始（上限: {limit if limit else '全て'}枚）")
            
            results = pipeline.process_batch(
                input_dir=data_dir,
                pattern='*.png',
                limit=limit
            )
            
            # 結果保存
            filename = f'quickstart_{len(results)}images.csv'
            pipeline.save_results(results, output_dir, filename)
            
            print(f"\n✅ 処理完了！")
            print(f"📁 結果保存: {output_dir / filename}")
        
        else:
            print("無効な選択です")
            return
        
        print("\n" + "=" * 70)
        print("✨ クイックスタート完了")
        print("=" * 70)
        print(f"\n次のステップ:")
        print(f"  1. {output_dir} で結果を確認")
        print(f"  2. notebooks/demo.ipynb でデモノートブックを試す")
        print(f"  3. config.yaml で設定をカスタマイズ")
        
    except KeyboardInterrupt:
        print("\n\n中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
