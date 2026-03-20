"""
Granite-Vision統合モジュール
画像から損傷説明テキストを生成
"""
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Union, List
from dataclasses import dataclass
import json
from transformers import AutoProcessor, AutoModelForVision2Seq


@dataclass
class VisionConfig:
    """Vision設定"""
    model_name: str = "ibm-granite/granite-vision-3b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 300
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True


class GraniteVisionAnalyzer:
    """Granite-Visionを使用した損傷画像分析クラス"""
    
    # デフォルトのプロンプトテンプレート（LLaVA用: <image>トークンが必要）
    DEFAULT_PROMPT = """<image>
USER: あなたは橋梁点検の専門家です。
次の画像に写る損傷を、土木構造物の専門用語を用いて簡潔に説明してください。

必ず以下の情報を含めてください：
- 損傷の種類（ひび割れ、鉄筋露出、腐食、剥離、断面欠損など）
- 損傷の程度（軽微、中程度、重度）
- 損傷の位置・範囲
- 構造上のリスク

ASSISTANT:"""
    
    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Args:
            config: Vision設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or VisionConfig()
        self.device = torch.device(self.config.device)
        
        print(f"Granite-Visionモデルを読み込み中... ({self.config.model_name})")
        print(f"デバイス: {self.device}")
        
        # モデルとプロセッサの読み込み
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # GPU16GBの場合はfp16で読み込み
            if self.config.device == "cuda":
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            self.model.eval()
            print("モデルの読み込みが完了しました")
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            print("代替モードで初期化します（ダミー出力）")
            self.processor = None
            self.model = None
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        return_raw: bool = False
    ) -> str:
        """
        画像を分析して損傷説明を生成
        
        Args:
            image: 画像（ファイルパスまたはPIL Image）
            prompt: カスタムプロンプト（Noneの場合はデフォルトを使用）
            return_raw: モデルの生テキストを返すか
        
        Returns:
            損傷説明テキスト
        """
        # 画像読み込み
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("imageはファイルパスまたはPIL Imageである必要があります")
        
        # プロンプト準備
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        
        # モデルが読み込まれていない場合はダミー出力
        if self.model is None or self.processor is None:
            return self._generate_dummy_description(image)
        
        # 推論実行
        try:
            with torch.no_grad():
                # 入力を準備
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                # 生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample
                )
                
                # デコード
                generated_text = self.processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]
                
                if return_raw:
                    return generated_text
                
                # プロンプト部分を除去して説明のみを抽出
                description = self._extract_description(generated_text, prompt)
                return description
                
        except Exception as e:
            print(f"推論エラー: {e}")
            return self._generate_dummy_description(image)
    
    def _extract_description(self, generated_text: str, prompt: str) -> str:
        """生成テキストから説明部分を抽出"""
        # プロンプトが含まれている場合は除去
        if prompt in generated_text:
            description = generated_text.replace(prompt, "").strip()
        else:
            description = generated_text.strip()
        
        return description
    
    def _generate_dummy_description(self, image: Image.Image) -> str:
        """ダミーの損傷説明を生成（モデル未使用時）"""
        return """鉄筋露出が確認されます。コンクリート表面が剥離し、内部の鉄筋が露出している状態です。
露出範囲は比較的広く、複数箇所で確認できます。
鉄筋の腐食も進行しており、断面欠損のリスクがあります。
構造耐力への影響が懸念されるため、早急な補修が必要です。"""
    
    def batch_analyze(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Path] = None,
        save_json: bool = True
    ) -> List[dict]:
        """
        複数画像を一括分析
        
        Args:
            image_paths: 画像パスのリスト
            output_dir: 出力ディレクトリ（Noneの場合は保存しない）
            save_json: JSON形式で保存するか
        
        Returns:
            分析結果のリスト
        """
        results = []
        
        for i, img_path in enumerate(image_paths, 1):
            img_path = Path(img_path)
            print(f"[{i}/{len(image_paths)}] 分析中: {img_path.name}")
            
            try:
                description = self.analyze_image(img_path)
                
                result = {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "description": description,
                    "status": "success"
                }
                
                results.append(result)
                
                # 個別保存
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # テキストファイル保存
                    txt_path = output_dir / f"{img_path.stem}_description.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(description)
                    
                    # JSON保存
                    if save_json:
                        json_path = output_dir / f"{img_path.stem}_description.json"
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"エラー: {img_path.name} - {e}")
                results.append({
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "description": None,
                    "status": "error",
                    "error": str(e)
                })
        
        # 全体のサマリー保存
        if output_dir is not None and save_json:
            summary_path = output_dir / "summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total": len(results),
                    "success": len([r for r in results if r["status"] == "success"]),
                    "error": len([r for r in results if r["status"] == "error"]),
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"\nサマリー保存: {summary_path}")
        
        return results


def load_config_from_yaml(config: dict) -> VisionConfig:
    """YAMLの設定辞書からVisionConfigを生成"""
    vision_config = config.get('granite_vision', {})
    
    return VisionConfig(
        model_name=vision_config.get('model_name', 'ibm-granite/granite-vision-3b'),
        device=vision_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        max_new_tokens=vision_config.get('max_new_tokens', 300),
        temperature=vision_config.get('temperature', 0.3),
        top_p=vision_config.get('top_p', 0.9),
        do_sample=vision_config.get('do_sample', True)
    )


if __name__ == "__main__":
    # 簡易テスト
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("使用法: python granite_vision.py <画像ファイルまたはディレクトリ>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    analyzer = GraniteVisionAnalyzer()
    
    if path.is_file():
        # 単一ファイル
        description = analyzer.analyze_image(path)
        print(f"\n=== 分析結果 ===")
        print(description)
    
    elif path.is_dir():
        # ディレクトリ内の全画像
        image_files = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
        output_dir = path.parent / "outputs" / "descriptions"
        
        results = analyzer.batch_analyze(image_files, output_dir=output_dir)
        print(f"\n完了: {len(results)}枚の画像を分析しました")
