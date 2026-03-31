"""
10789枚の損傷画像に対する推論実行スクリプト
完成したv0.2パイプラインを使用して大規模データセットの推論を実行
"""
import os
import sys
import io
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm

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


def save_results_with_ground_truth(results, ground_truth_map, output_dir, filename):
    """
    結果を保存（正解データ付き）
    
    Args:
        results: パイプライン処理結果のリスト
        ground_truth_map: {ファイル名: 所見} のマッピング辞書
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV用のデータを準備
    csv_data = []
    json_data = []
    
    for result in results:
        filename_only = Path(result.image_path).name
        
        # 基本情報
        row = {
            'ファイル名': filename_only,
            '画像パス': result.image_path,
            'ステータス': result.status,
            '処理時間_秒': round(result.processing_time, 2) if result.processing_time else None,
        }
        
        # 正解データを追加
        row['所見_正解'] = ground_truth_map.get(filename_only, '')
        
        if result.status == 'success':
            # 損傷説明
            row['損傷説明'] = result.description
            
            # 構造化データ
            row['損傷種別'] = result.structure.get('damage_type', '') if result.structure else ''
            row['重症度'] = result.structure.get('severity', '') if result.structure else ''
            row['位置'] = result.structure.get('location', '') if result.structure else ''
            row['範囲'] = result.structure.get('extent', '') if result.structure else ''
            row['リスク'] = result.structure.get('risk', '') if result.structure else ''
            
            # スコアリング結果
            row['生スコア'] = round(result.score.get('raw_score', 0), 3) if result.score else 0
            row['優先度レベル'] = result.score.get('priority_level', '') if result.score else ''
            row['優先度説明'] = result.score.get('priority_description', '') if result.score else ''
            row['損傷種別スコア'] = round(result.score.get('damage_type_score', 0), 3) if result.score else 0
            row['重症度スコア'] = round(result.score.get('severity_score', 0), 3) if result.score else 0
            row['位置スコア'] = round(result.score.get('location_score', 0), 3) if result.score else 0
            row['リスクスコア'] = round(result.score.get('risk_score', 0), 3) if result.score else 0
            
            # JSONデータに詳細を追加
            json_record = {
                'filename': filename_only,
                'image_path': result.image_path,
                'ground_truth': ground_truth_map.get(filename_only, ''),
                'status': result.status,
                'processing_time': result.processing_time,
                'description': result.description,
                'structure': result.structure,
                'score': result.score
            }
            json_data.append(json_record)
        else:
            # エラー情報
            row['エラー'] = result.error if hasattr(result, 'error') else 'Unknown error'
        
        csv_data.append(row)
    
    # CSVとして保存
    csv_file = output_dir / filename
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"💾 CSV保存: {csv_file}")
    
    # JSONとして保存
    json_file = output_dir / filename.replace('.csv', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON保存: {json_file}")
    
    # 統計情報を表示
    print(f"\n📊 処理統計:")
    print(f"  総画像数: {len(results)}枚")
    success_count = sum(1 for r in results if r.status == 'success')
    print(f"  成功: {success_count}枚 ({success_count/len(results)*100:.1f}%)")
    error_count = len(results) - success_count
    if error_count > 0:
        print(f"  失敗: {error_count}枚 ({error_count/len(results)*100:.1f}%)")
    
    if success_count > 0:
        avg_time = sum(r.processing_time for r in results if r.status == 'success') / success_count
        print(f"  平均処理時間: {avg_time:.2f}秒/枚")
        total_time = sum(r.processing_time for r in results if r.status == 'success')
        print(f"  総処理時間: {total_time/60:.1f}分 ({total_time/3600:.2f}時間)")


def main():
    """メイン処理"""
    print("=" * 80)
    print("Bridge Damage Assessment - 大規模推論実行 (N=10789)")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # パス設定
    master_file = project_root / 'data' / 'image_text_inspect_n10789' / 'master' / 'Rank_c_image_text_n10789.xlsx'
    # 前処理済み画像を使用（IMAGE_DIR_OVERRIDEで上書き可能）
    image_dir_override = os.getenv('IMAGE_DIR_OVERRIDE')
    image_dir = Path(image_dir_override) if image_dir_override else (project_root / 'data' / 'preprocessed_640x480_n10789')
    output_dir = project_root / 'data' / 'outputs'
    config_path = project_root / 'config.yaml'
    
    # 前処理済みディレクトリの確認
    if not image_dir.exists():
        print(f"❌ 前処理済み画像ディレクトリが見つかりません: {image_dir}")
        print(f"   先に preprocess_n10789.py を実行してください")
        return
    
    # マスターデータ読み込み
    try:
        print(f"📁 マスターデータ読み込み中: {master_file}")
        master_df = pd.read_excel(master_file)
        print(f"✅ マスターデータ読み込み完了: {len(master_df)}行")
        
        # 正解データのマッピングを作成（ファイルパス -> 所見）
        master_filtered = master_df[['ファイルパス', '所見']].dropna(subset=['ファイルパス'])
        ground_truth_map = {}
        for _, row in master_filtered.iterrows():
            filename = str(row['ファイルパス'])
            gt = str(row['所見']) if pd.notna(row['所見']) else ''
            ground_truth_map[filename] = gt
        
        print(f"📊 正解データ: {len(ground_truth_map)}件")
    except Exception as e:
        print(f"❌ マスターデータ読み込みエラー: {e}")
        return
    
    # 画像ファイル確認
    image_files = sorted(image_dir.glob('*.jpg'), key=lambda p: p.name)  # 再現性のためソート
    print(f"\n📂 画像ディレクトリ: {image_dir}")
    print(f"📊 画像ファイル数: {len(image_files)}枚")

    # 未処理リスト指定（TARGET_FILELIST_CSV）
    # CSVは「ファイル名」列を優先し、なければ先頭列を使用
    target_filelist_csv = os.getenv('TARGET_FILELIST_CSV')
    if target_filelist_csv:
        target_path = Path(target_filelist_csv)
        if not target_path.exists():
            print(f"❌ TARGET_FILELIST_CSVが見つかりません: {target_path}")
            return
        try:
            target_df = pd.read_csv(target_path, encoding='utf-8-sig')
            if target_df.empty:
                print(f"❌ TARGET_FILELIST_CSVが空です: {target_path}")
                return

            target_column = 'ファイル名' if 'ファイル名' in target_df.columns else target_df.columns[0]
            target_names = set(target_df[target_column].dropna().astype(str).str.strip())
            before_count = len(image_files)
            image_files = [f for f in image_files if f.name in target_names]
            print(f"📌 TARGET_FILELIST_CSV指定: {target_path}")
            print(f"📊 対象ファイル抽出: {before_count}枚 -> {len(image_files)}枚")
        except Exception as e:
            print(f"❌ TARGET_FILELIST_CSV読み込みエラー: {e}")
            return
    
    if len(image_files) == 0:
        print("❌ 処理対象の画像が見つかりません")
        return
    
    # 再開位置の決定（優先順: RESUME_FROM_COUNT > チェックポイント）
    resume_offset = 0
    resume_from_count = os.getenv('RESUME_FROM_COUNT')
    if resume_from_count:
        try:
            resume_offset = int(resume_from_count)
            if resume_offset < 0:
                resume_offset = 0
            if resume_offset > len(image_files):
                resume_offset = len(image_files)
            print(f"\n📌 RESUME_FROM_COUNT指定: {resume_offset}枚目までをスキップ")
            image_files = image_files[resume_offset:]
            print(f"📊 残り処理対象: {len(image_files)}枚")
        except ValueError:
            print(f"⚠️ RESUME_FROM_COUNTの値が不正です: {resume_from_count}")
            return
    else:
        # チェックポイントから再開（処理済みファイルをスキップ）
        checkpoint_files = sorted(
            output_dir.glob('inference_n10789_checkpoint_*.csv'),
            key=lambda p: p.name
        )
        processed_filenames = set()
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"\n📂 チェックポイント検出: {latest_checkpoint.name}")
            try:
                ckpt_df = pd.read_csv(latest_checkpoint, encoding='utf-8-sig')
                processed_filenames = set(ckpt_df['ファイル名'].dropna().astype(str))
                resume_offset = len(processed_filenames)
                print(f"✅ 処理済み: {resume_offset}枚 → スキップします")
            except Exception as e:
                print(f"⚠️ チェックポイント読み込みエラー: {e}")
        
        if processed_filenames:
            image_files = [f for f in image_files if f.name not in processed_filenames]
            print(f"📊 残り処理対象: {len(image_files)}枚")
    
    if len(image_files) == 0:
        print("✅ すべての画像が処理済みです")
        return
    
    # 処理範囲の確認
    print(f"\n⚠️  注意: {len(image_files)}枚の画像を処理します")
    print(f"予想処理時間: 約{len(image_files) * 8.2 / 60:.1f}分 ({len(image_files) * 8.2 / 3600:.2f}時間)")
    print(f"\n自動的に処理を開始します...")
    
    # パイプライン初期化
    print(f"\n🔧 パイプライン初期化中...")
    try:
        pipeline = DamageAnalysisPipeline(str(config_path))
    except Exception as e:
        print(f"❌ パイプライン初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 推論実行（前処理済み画像のためpreprocess=False）
    print(f"\n🚀 推論開始...")
    print(f"   前処理済み画像を使用するため、リサイズ処理はスキップします")
    start_time = datetime.now()
    
    results = []
    
    try:
        for i, img_path in enumerate(tqdm(image_files, desc="画像処理中"), 1):
            result = pipeline.process_image(
                img_path,
                preprocess=False,  # 前処理済み画像なのでスキップ
                save_intermediate=False
            )
            results.append(result)
            absolute_done = resume_offset + i
            
            # 中間保存（累積1000枚ごと）
            if absolute_done % 1000 == 0:
                print(f"\n💾 中間保存中（累積{absolute_done}枚完了）...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_results_with_ground_truth(
                    results, 
                    ground_truth_map,
                    output_dir, 
                    f'inference_n10789_checkpoint_{absolute_done}_{timestamp}.csv'
                )
    except KeyboardInterrupt:
        print("\n\n⚠️  処理が中断されました")
        if len(results) > 0:
            print(f"💾 中断時点のデータを保存します（{len(results)}枚）...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_results_with_ground_truth(
                results, 
                ground_truth_map,
                output_dir, 
                f'inference_n10789_interrupted_{timestamp}.csv'
            )
        return
    except Exception as e:
        print(f"\n❌ 処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 結果保存（正解データ付き）
    print(f"\n💾 結果保存中...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results_with_ground_truth(
        results, 
        ground_truth_map,
        output_dir, 
        f'inference_n10789_{timestamp}.csv'
    )
    
    # 完了
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n" + "=" * 80)
    print("✨ 処理完了")
    print("=" * 80)
    print(f"総処理時間: {total_time/60:.1f}分 ({total_time/3600:.2f}時間)")
    if len(results) > 0:
        print(f"平均処理時間: {total_time/len(results):.2f}秒/枚")
    print(f"出力先: {output_dir}")
    print()


if __name__ == "__main__":
    main()
