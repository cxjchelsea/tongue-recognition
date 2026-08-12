from pathlib import Path
import pandas as pd
from .base import DatasetAdapter
from ..schema import SampleRecord
from ..utils import normalize_na

class TongueDxAdapter(DatasetAdapter):
    """TongueDx：支持多 CSV（fold/split）与独立图像根目录。"""

    def _csv_specs(self):
        # 兼容旧配置 csv: 单文件；新配置 csv_files: [{path, source_split}]
        if self.cfg.get("csv_files"):
            return list(self.cfg["csv_files"])
        if self.cfg.get("csv"):
            return [{"path": self.cfg["csv"], "source_split": self.cfg.get("source_split")}]
        return []

    def collect(self):
        samples, labels, spatial, warnings = [], [], [], []
        specs = self._csv_specs()
        if not specs:
            return samples, labels, spatial, [f"{self.name}: csv / csv_files missing"]

        images_root = self.root / self.cfg.get("images_root", ".")
        image_col = self.cfg.get("image_path_column", "image_path")
        patient_col = self.cfg.get("patient_id_column", "id")
        seen = set()

        for spec in specs:
            rel = spec["path"] if isinstance(spec, dict) else spec
            source_split = spec.get("source_split") if isinstance(spec, dict) else None
            csv_path = self.root / rel
            if not csv_path.exists():
                warnings.append(f"{self.name}: csv missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                raw = normalize_na(row.get(image_col))
                if raw is None:
                    continue
                img = Path(str(raw))
                if not img.is_absolute():
                    # 优先按 images_root 解析；兼容已含 origin/ 前缀的路径
                    candidate = images_root / img
                    img = candidate if candidate.exists() else (self.root / img)
                if not img.exists():
                    warnings.append(f"{self.name}: image missing: {img}")
                    continue
                source_id = str(img.relative_to(self.root)).replace("\\", "/") if self.root in img.parents else img.name
                sid = self.make_sample_id(source_id)
                if sid not in seen:
                    md5, wid, hei = self.basic_image_meta(img)
                    patient = normalize_na(row.get(patient_col))
                    samples.append(SampleRecord(
                        sid, self.name, source_id, str(img), md5, wid, hei,
                        patient_id=None if patient is None else str(patient),
                        patient_id_available=patient is not None,
                        source_split=source_split,
                    ))
                    seen.add(sid)
                for col in self.cfg.get("label_columns", []):
                    if col not in row.index:
                        continue
                    value = normalize_na(row.get(col))
                    if value is None:
                        continue
                    recs, warn_list = self.mapping_to_label_records(sid, col, col, value=value)
                    labels.extend(recs)
                    warnings.extend(warn_list)
        return samples, labels, spatial, warnings
