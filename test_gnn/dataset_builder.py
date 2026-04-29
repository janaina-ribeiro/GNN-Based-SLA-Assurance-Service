from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphSplit:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset


def temporal_split(
    dataset: "DelayGraphDataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> GraphSplit:
    """
    Temporal train/val/test split based on sample timestamps.
    -----------------------------------------------------------
    The dataset is ordered by its internal timestamp vector and then split
    sequentially according to the provided ratios, mirroring the strategy
    used for the GNN experiments so it can be reused for other models.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be >= 0 and train_ratio + val_ratio < 1")

    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Cannot split empty dataset")

    timestamps = np.array(dataset.sample_timestamps)
    order = np.argsort(timestamps)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test <= 0:
        raise ValueError(
            f"Split ratios too aggressive for {n_samples} samples: "
            f"train={train_ratio}, val={val_ratio}."
        )

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    return GraphSplit(
        train=Subset(dataset, train_idx.tolist()),
        val=Subset(dataset, val_idx.tolist()),
        test=Subset(dataset, test_idx.tolist()),
    )


def _infer_step_minutes(timestamps: pd.Series) -> float:
    if timestamps.empty:
        return 5.0
    ordered = timestamps.sort_values().drop_duplicates()
    deltas = ordered.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 5.0
    return max(float(np.median(deltas)), 1.0)


def _extract_path_features(df: pd.DataFrame) -> Dict[str, float]:

    """
    Extracts traceroute features from the DataFrame.
    ---------------------------------------------------

    Extracted features (simplified):
    - avg_hops: average number of hops
    - min_hops: minimum number of hops
    - max_hops: maximum number of hops
    - std_hops: standard deviation of the number of hops

    Args:
        df: DataFrame with traceroute data

    Returns:
        Dict with the 4 traceroute features
    """
    if df.empty or "Num_Hops" not in df.columns:
        return {"avg_hops": 0.0, "min_hops": 0.0, "max_hops": 0.0, "std_hops": 0.0}

    hops = df["Num_Hops"].fillna(0).astype("float32")
    if hops.empty or hops.isna().all():
        return {"avg_hops": 0.0, "min_hops": 0.0, "max_hops": 0.0, "std_hops": 0.0}

    avg_hops = float(hops.mean())
    min_hops = float(hops.min())
    max_hops = float(hops.max())
    std_hops = float(hops.std()) if len(hops) > 1 else 0.0

    return {
        "avg_hops": avg_hops,
        "min_hops": min_hops,
        "max_hops": max_hops,
        "std_hops": std_hops,
    }


def _calculate_path_similarity(path1: str, path2: str) -> float:
    """
    Calculates similarity between two IP paths.
    ------------------------------------------
    Args:
        path1, path2: Path strings in the format "IP1 -> IP2 -> IP3"

    Returns:
        Jaccard similarity between the sets of IPs
    """
    if not path1 or not path2 or pd.isna(path1) or pd.isna(path2):
        return 0.0

    try:
        ips1 = set(path1.split(" -> "))
        ips2 = set(path2.split(" -> "))

        if not ips1 or not ips2:
            return 0.0

        intersection = len(ips1.intersection(ips2))
        union = len(ips1.union(ips2))

        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _compute_topology_features(
    frames: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    
    """
    Computes topological features for each link.
    ------------------------------------------
    Args:
        frames: Dict with DataFrames per link

    Returns:
        Dict with topological features per link
    """
    topo_features = {}

    for link, df in frames.items():
        topo_features[link] = _extract_path_features(df)

    return topo_features


def _resample_frame(
    df: pd.DataFrame, freq_minutes: int, delay_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Resamples the dataframe to a uniform frequency.
    -----------------------------------------------
    Preserves numeric columns such as Atraso(ms) and Num_Hops after resampling.

    Args:
        df: DataFrame with columns Timestamp, Atraso(ms) and optionally Num_Hops
        freq_minutes: Resampling frequency in minutes
        delay_threshold: Fixed threshold for high delay(ms). If None, will be calculated by percentile later.
    """
    frame = df.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=False)
    frame = frame.sort_values("Timestamp")
    frame = frame.set_index("Timestamp")
    rule = f"{int(freq_minutes)}min"

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()

    agg = frame[numeric_cols].resample(rule).mean()

    agg["Atraso"] = agg["Atraso"].interpolate(
        method="linear", limit_direction="both"
    )
    agg["Atraso"] = agg["Atraso"].ffill().bfill()

    if "Num_Hops" in agg.columns:
        agg["Num_Hops"] = agg["Num_Hops"].interpolate(
            method="linear", limit_direction="both"
        )
        agg["Num_Hops"] = agg["Num_Hops"].ffill().bfill()

    if delay_threshold is not None:
        agg["high_delay"] = (agg["Atraso"] > delay_threshold).astype(int)
    agg = agg.dropna(subset=["Atraso"], how="all")
    agg = agg.reset_index()
    return agg


def _build_feature_vector(
    delay_window: pd.Series,
    hops_window: Optional[pd.Series] = None,
    global_topo_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Builds a feature vector combining delay(ms) and traceroute data.
    --------------------------------------------
    Delay features (5 features) - calculated PER WINDOW:
    - mean_delay: mean delay(ms) in the window
    - std_delay: standard deviation of delay(ms)
    - max_delay: maximum delay(ms) in the window
    - min_delay: minimum delay(ms) in the window
    - last_delay: last delay(ms) value (most recent)

    Traceroute features (4 features) - calculated PER WINDOW when available:
    - avg_hops: mean number of hops in the window
    - min_hops: minimum number of hops in the window
    - max_hops: maximum number of hops in the window
    - std_hops: standard deviation of hops in the window (indicates route instability)

    If hops_window is not available, uses global_topo_features as fallback.

    Args:
        delay_window: Time series of delay(ms) in the window
        hops_window: Time series of hops in the window (optional)
        global_topo_features: Global traceroute features (fallback)

    Returns:
        Array of combined features (5 or 9 features depending on traceroute)
    """
    values = delay_window.to_numpy(dtype=np.float32)

    mean_delay = float(np.nanmean(values))
    std_delay = float(np.nanstd(values)) if len(values) > 1 else 0.0
    max_delay = float(np.nanmax(values))
    min_delay = float(np.nanmin(values))
    last_delay = float(values[-1]) if not np.isnan(values[-1]) else mean_delay

    delay_features = np.array(
        [mean_delay, std_delay, max_delay, min_delay, last_delay], dtype=np.float32
    )

    if hops_window is not None and len(hops_window) > 0:
        hops_values = hops_window.to_numpy(dtype=np.float32)
        hops_values = hops_values[~np.isnan(hops_values)]

        if len(hops_values) > 0:
            avg_hops = float(np.mean(hops_values))
            min_hops = float(np.min(hops_values))
            max_hops = float(np.max(hops_values))
            std_hops = float(np.std(hops_values)) if len(hops_values) > 1 else 0.0
        else:
            avg_hops = min_hops = max_hops = std_hops = 0.0

        topo_array = np.array(
            [avg_hops, min_hops, max_hops, std_hops], dtype=np.float32
        )
        feat = np.concatenate([delay_features, topo_array])

    elif global_topo_features is not None:
        topo_array = np.array(
            [
                global_topo_features.get("avg_hops", 0.0),
                global_topo_features.get("min_hops", 0.0),
                global_topo_features.get("max_hops", 0.0),
                global_topo_features.get("std_hops", 0.0),
            ],
            dtype=np.float32,
        )
        feat = np.concatenate([delay_features, topo_array])
    else:
        feat = delay_features

    return feat.astype(np.float32)


def _build_edge_index(
    delays: pd.DataFrame,
    min_corr: float,
    topo_features: Optional[Dict[str, Dict[str, float]]] = None,
    path_similarity: Optional[Dict[Tuple[str, str], float]] = None,
    alpha: float = 0.6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Builds edge indices combining delay correlation and topological similarity.
    ------------------------------------------
    Args:
        delays: DataFrame with delay(ms) columns per link
        min_corr: Minimum correlation to create edge
        topo_features: Topological features per link
        path_similarity: Path similarity matrix
        alpha: Weight for delay correlation (1-alpha for topology)

    Returns:
        Tuple with edge_index and edge_weights
    """
    corr = delays.corr(method="spearman").fillna(0.0)
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    columns = list(corr.columns)

    link_names = [col.replace("delay_", "") for col in columns]

    for i, src in enumerate(columns):
        for j, dst in enumerate(columns):
            if i == j:
                continue

            delay_corr = float(abs(corr.loc[src, dst]))

            topo_sim = 0.0
            if topo_features and path_similarity:
                link_i = link_names[i]
                link_j = link_names[j]

                path_sim = path_similarity.get((link_i, link_j), 0.0)

                if link_i in topo_features and link_j in topo_features:
                    feat_i = topo_features[link_i]
                    feat_j = topo_features[link_j]

                    hop_diff = abs(
                        feat_i.get("avg_hops", 0) - feat_j.get("avg_hops", 0)
                    )
                    hop_sim = 1.0 / (1.0 + hop_diff) if hop_diff > 0 else 1.0

                    std_diff = abs(
                        feat_i.get("std_hops", 0) - feat_j.get("std_hops", 0)
                    )
                    std_sim = 1.0 / (1.0 + std_diff) if std_diff > 0 else 1.0

                    topo_sim = 0.5 * path_sim + 0.3 * hop_sim + 0.2 * std_sim

            combined_weight = alpha * delay_corr + (1 - alpha) * topo_sim

            if combined_weight >= min_corr:
                edges.append((i, j))
                weights.append(combined_weight)

    if not edges:
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i == j:
                    continue
                edges.append((i, j))
                weights.append(0.1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class DelayGraphDataset(Dataset):


    def __init__(
        self,
        data_dir: Path | str,
        links: Optional[Sequence[str]] = None,
        window_size: int = 3,
        horizon_minutes: int = 15,
        min_corr: float = 0.3,
        limit_samples: Optional[int] = None,
        delay_threshold: Optional[float] = None,
        delay_percentile: float = 85.0,
        topology_weight: float = 0.4,
        column_delay: str = "Atraso",
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.root = Path(data_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root}")
        self.links = self._discover_links(links)
        self.topology_weight = topology_weight
        self.column_delay = column_delay
        print(
            f"[DEBUG] Found {len(self.links)} links: {self.links[:3]}{'...' if len(self.links) > 3 else ''}"
        )
        print(f"[DEBUG] Traceroute features: always enabled (mandatory)")
        raw_frames = self._load_frames()
        print(f"[DEBUG] Loaded {len(raw_frames)} raw frames (showing first 2)")
        for link, df in list(raw_frames.items())[:2]:
            print(f"[DEBUG]   - {link}: {len(df)} rows")

        self.topo_features = _compute_topology_features(raw_frames)
        self.path_similarity = self._compute_path_similarity_matrix(raw_frames)
        print(
            f"[DEBUG] Computed topology features for {len(self.topo_features)} links"
        )
        freq_minutes = (
            int(
                round(
                    np.median(
                        [
                            _infer_step_minutes(f["Timestamp"])
                            for f in raw_frames.values()
                        ]
                    )
                )
            )
            or 5
        )
        print(f"[DEBUG] Detected frequency: {freq_minutes} minutes")

        if delay_threshold is None:
            all_delays = pd.concat(
                [df[self.column_delay] for df in raw_frames.values()], ignore_index=True
            )
            delay_threshold = float(all_delays.quantile(delay_percentile / 100.0))
            print(
                f"[DEBUG] Delay threshold (percentile {delay_percentile}%): {delay_threshold:.2f} ms"
            )
        else:
            print(f"[DEBUG] Using fixed delay threshold: {delay_threshold:.2f} ms")

        self.delay_threshold = delay_threshold
        self.delay_percentile = delay_percentile

        print("[DEBUG] Resampling data...")
        resampled = {
            link: _resample_frame(df, freq_minutes, delay_threshold, self.column_delay)
            for link, df in raw_frames.items()
        }
        total_resampled = sum(len(df) for df in resampled.values())
        print(f"[DEBUG] Resampling complete: {total_resampled} total rows")
        combined = self._combine(resampled)
        offset_steps = max(int(round(horizon_minutes / freq_minutes)), 1)
        print(
            f"[DEBUG] offset_steps = {offset_steps} (horizon={horizon_minutes}min / freq={freq_minutes}min)"
        )
        node_features, node_labels, timestamps = self._create_samples(
            combined, window_size, offset_steps, limit_samples
        )
        self.features = torch.from_numpy(node_features)
        self.labels = torch.from_numpy(node_labels).long()
        delay_cols = [f"delay_{link}" for link in self.links]
        edge_index, edge_weight = _build_edge_index(
            combined[delay_cols],
            min_corr,
            self.topo_features,
            self.path_similarity,
            alpha=1.0 - self.topology_weight,
        )
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.sample_timestamps = timestamps
        self._combined_frame = combined
        self._freq_minutes = freq_minutes
        self._window_size = window_size
        self._offset_steps = offset_steps


    def _to_serializable(self) -> Dict[str, object]:
        """Convert internal tensors and metadata to a joblib-friendly dict."""
        return {
            "features": self.features.cpu().numpy(),
            "labels": self.labels.cpu().numpy(),
            "edge_index": self.edge_index.cpu().numpy(),
            "edge_weight": self.edge_weight.cpu().numpy(),
            "sample_timestamps": [ts.isoformat() for ts in self.sample_timestamps],
            "links": list(self.links),
            "delay_threshold": float(self.delay_threshold),
            "delay_percentile": float(self.delay_percentile),
            "topology_weight": float(self.topology_weight),
            "column_delay": str(self.column_delay),
            "freq_minutes": int(self._freq_minutes),
            "window_size": int(self._window_size),
            "offset_steps": int(self._offset_steps),
        }

    @classmethod
    def _from_serializable(cls, payload: Dict[str, object]) -> "DelayGraphDataset":
        """Rebuild a DelayGraphDataset-like instance from a serialized dict.

        This bypasses the __init__ that reads CSVs and instead restores tensors
        and metadata directly, so it can be used for training without
        reprocessing the raw data.
        """
        obj = cls.__new__(cls)   
        Dataset.__init__(obj)

        obj.features = torch.from_numpy(payload["features"]).float() 
        obj.labels = torch.from_numpy(payload["labels"]).long()  
        obj.edge_index = torch.from_numpy(payload["edge_index"]).long()  
        obj.edge_weight = torch.from_numpy(payload["edge_weight"]).float()  

        obj.sample_timestamps = [
            pd.Timestamp(ts) for ts in payload["sample_timestamps"]  

        ]
        obj.links = list(payload["links"])  

        obj.delay_threshold = float(payload["delay_threshold"])  
        obj.delay_percentile = float(payload["delay_percentile"])  
        obj.topology_weight = float(payload.get("topology_weight", 0.4))  
        obj.column_delay = str(payload.get("column_delay", "Atraso"))  

        obj._freq_minutes = int(payload.get("freq_minutes", 5))  
        obj._window_size = int(payload.get("window_size", 3))  
        obj._offset_steps = int(payload.get("offset_steps", 1))  

        obj._combined_frame = None
        return obj

    def save_joblib(self, output_path: Path | str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._to_serializable(), path)

    @classmethod
    def load_joblib(cls, input_path: Path | str) -> "DelayGraphDataset":
        payload = joblib.load(Path(input_path))
        if not isinstance(payload, dict):
            raise ValueError("Invalid joblib payload: expected a dict")
        return cls._from_serializable(payload)

    def _discover_links(self, links: Optional[Sequence[str]]) -> List[str]:
        if links:
            return list(links)
        candidates = sorted(
            path.name[len("dataset_") : -len("_links_hops.csv")]
            for path in self.root.glob("dataset_*_links_hops.csv")
        )
        if not candidates:
            raise FileNotFoundError("No dataset_*_links_hops.csv files were found")

        print(
            f"[DEBUG] Found {len(candidates)} links, using all for complete processing"
        )
        return candidates

    def _load_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for link in self.links:
            file_path = self.root / f"dataset_{link}_links_hops.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file for link {link}: {file_path}")
            df = pd.read_csv(file_path, parse_dates=["Timestamp"])
            if self.column_delay not in df.columns:
                raise ValueError(f"Column '{self.column_delay}' not found in {file_path}")

            if "Num_Hops" not in df.columns and "Total_Hops" not in df.columns:
                raise ValueError(f"Column 'Num_Hops' or 'Total_Hops' not found in {file_path}, traceroute features are mandatory.")
            if "Total_Hops" in df.columns and "Num_Hops" not in df.columns:
                df = df.rename(columns={"Total_Hops": "Num_Hops"})
            if "Path_IPs" not in df.columns:
                raise ValueError(f"Column 'Path_IPs' not found in {file_path}, traceroute features are mandatory.")

            required_cols = ["Timestamp", self.column_delay, "Num_Hops", "Path_IPs"]
            if "Path_Hostnames" in df.columns:
                required_cols.append("Path_Hostnames")
            df = df[[col for col in required_cols if col in df.columns]]

            df = df.dropna(subset=["Timestamp", self.column_delay])
            frames[link] = df
        return frames

    def _compute_path_similarity_matrix(
        self, frames: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], float]:
        """
        Computes a similarity matrix between paths of different links.
        -------------------------------------------------------------------
        Optimization: Calculates only the upper half of the matrix (i < j) and mirrors,
        reducing complexity from O(n²) to O(n²/2).

        Args:
            frames: Dict with DataFrames per link

        Returns:
            Dict with similarity between pairs of links
        """
        similarity_matrix = {}
        links = list(frames.keys())
        n_links = len(links)

        for link in links:
            similarity_matrix[(link, link)] = 1.0

        for i in range(n_links):
            link1 = links[i]
            df1 = frames[link1]
            has_path1 = "Path_IPs" in df1.columns
            paths1 = df1["Path_IPs"].dropna().unique() if has_path1 else []

            for j in range(i + 1, n_links):
                link2 = links[j]
                df2 = frames[link2]
                has_path2 = "Path_IPs" in df2.columns

                if not has_path1 or not has_path2:
                    similarity_matrix[(link1, link2)] = 0.0
                    similarity_matrix[(link2, link1)] = 0.0
                    continue

                paths2 = df2["Path_IPs"].dropna().unique()

                if len(paths1) == 0 or len(paths2) == 0:
                    similarity_matrix[(link1, link2)] = 0.0
                    similarity_matrix[(link2, link1)] = 0.0
                    continue

                similarities = []
                sample_size = min(3, len(paths1), len(paths2))
                for p1 in paths1[:sample_size]:
                    for p2 in paths2[:sample_size]:
                        sim = _calculate_path_similarity(p1, p2)
                        similarities.append(sim)

                avg_similarity = np.mean(similarities) if similarities else 0.0
                similarity_matrix[(link1, link2)] = float(avg_similarity)
                similarity_matrix[(link2, link1)] = float(avg_similarity)

        return similarity_matrix

    def _combine(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        delay_frames = []
        hops_frames = []

        for link, frame in frames.items():
            clean_frame = frame[["Timestamp", self.column_delay]].copy()
            ren = clean_frame.rename(columns={self.column_delay: f"delay_{link}"})
            ren = ren.set_index("Timestamp")
            delay_frames.append(ren)

            if "Num_Hops" in frame.columns:
                hops_frame = frame[["Timestamp", "Num_Hops"]].copy()
                hops_ren = hops_frame.rename(columns={"Num_Hops": f"hops_{link}"})
                hops_ren = hops_ren.set_index("Timestamp")
                hops_frames.append(hops_ren)

        merged = pd.concat(delay_frames, axis=1, join="outer")

        if hops_frames:
            hops_merged = pd.concat(hops_frames, axis=1, join="outer")
            merged = pd.concat([merged, hops_merged], axis=1)

        merged = merged.sort_index()
        merged = merged.ffill().bfill()
        merged = merged.interpolate(method="time", limit_direction="both")

        target_data = {}
        for link in self.links:
            delay_col = f"delay_{link}"
            target_data[f"target_{link}"] = (
                merged[delay_col] > self.delay_threshold
            ).astype(int)

        target_df = pd.DataFrame(target_data, index=merged.index)
        merged = pd.concat([merged, target_df], axis=1)

        merged = merged.fillna(0)
        merged = merged.reset_index()

        total_targets = sum(
            (merged[f"target_{link}"] == 1).sum() for link in self.links
        )
        total_samples = len(merged) * len(self.links)
        print(f"[DEBUG] Combined frame has {len(merged)} rows after processing")
        print(
            f"[DEBUG] Class distribution: {total_targets} high delay ({100 * total_targets / total_samples:.1f}%) vs {total_samples - total_targets} normal ({100 * (total_samples - total_targets) / total_samples:.1f}%)"
        )

        return merged

    def _create_samples(
        self,
        combined: pd.DataFrame,
        window_size: int,
        offset_steps: int,
        limit_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        delay_cols = [f"delay_{link}" for link in self.links]
        hops_cols = [f"hops_{link}" for link in self.links]
        target_cols = [f"target_{link}" for link in self.links]

        available_hops_cols = {col: col in combined.columns for col in hops_cols}
        has_hops_data = any(available_hops_cols.values())

        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        stamps: List[pd.Timestamp] = []
        total_rows = len(combined)

        print(
            f"[DEBUG] Creating samples: total_rows={total_rows}, window_size={window_size}, offset_steps={offset_steps}"
        )
        print(
            f"[DEBUG] Hops data available: {has_hops_data} ({sum(available_hops_cols.values())}/{len(hops_cols)} links)"
        )

        min_required = window_size + offset_steps + 1
        if total_rows < min_required:
            raise ValueError(
                f"Not enough data: have {total_rows} rows but need at least {min_required} "
                f"(window_size={window_size} + offset_steps={offset_steps} + 1). "
                f"Try reducing window_size or horizon_minutes, or check your data."
            )

        for idx in range(window_size, total_rows - offset_steps):
            window_start = idx - window_size
            window_end = idx
            target_pos = idx + offset_steps - 1

            if target_pos >= total_rows:
                break

            window_data = combined.iloc[window_start:window_end]
            target_row = combined.iloc[target_pos]

            node_feat_list = []
            for link_idx, delay_col in enumerate(delay_cols):
                link_name = self.links[link_idx]
                hops_col = hops_cols[link_idx]

                delay_window = pd.Series(window_data[delay_col].values)

                hops_window = None
                if available_hops_cols.get(hops_col, False):
                    hops_window = pd.Series(window_data[hops_col].values)

                global_topo = (
                    self.topo_features.get(link_name) if self.topo_features else None
                )

                feat_vector = _build_feature_vector(
                    delay_window, hops_window, global_topo
                )
                node_feat_list.append(feat_vector)

            node_feat = np.stack(node_feat_list)
            node_label = target_row[target_cols].to_numpy(dtype=np.int64)
            features.append(node_feat)
            labels.append(node_label)
            stamps.append(pd.Timestamp(target_row["Timestamp"]))

            if limit_samples and len(features) >= limit_samples:
                break

            if len(features) % 1000 == 0:
                print(f"[DEBUG] Processed {len(features)} samples...")

        print(f"[DEBUG] Generated {len(features)} samples")

        if not features:
            raise ValueError(
                f"No samples generated with {total_rows} rows, window_size={window_size}, offset_steps={offset_steps}. "
                f"Consider lowering window_size or horizon_minutes."
            )
        feature_array = np.stack(features)
        label_array = np.stack(labels)
        return feature_array, label_array, stamps

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Data:
        x = self.features[idx]
        y = self.labels[idx]
        data = Data(
            x=x,
            edge_index=self.edge_index.clone(),
            y=y,
        )
        data.edge_weight = self.edge_weight.clone()
        data.timestamp = self.sample_timestamps[idx]
        return data

    @property
    def num_node_features(self) -> int:
        return int(self.features.shape[-1])

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[1])

    @property
    def link_order(self) -> List[str]:
        return list(self.links)

    def export_combined_frame(self, output_path: Path | str) -> None:
        """
        Exports the combined temporal DataFrame (before windowing) to CSV.
        -----------------------------------------------------------------
        This allows tracking the delay(ms) series per link and the future target
        generated for each timestamp.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._combined_frame.to_csv(path, index=False)

    def export_window_samples(
        self, output_path: Path | str, limit: Optional[int] = 100
    ) -> None:
        """
        Exports a summary of the first windows (features and labels) for inspection.
        -----------------------------------------------------------------
        The file contains, for each sample and link, the target timestamp and the associated label
        (0/1), as well as the sample index. The full features may be large, so they are not expanded here.
        """
        import pandas as pd

        max_samples = min(
            len(self.sample_timestamps), limit or len(self.sample_timestamps)
        )
        records = []
        for i in range(max_samples):
            ts = self.sample_timestamps[i]
            labels = self.labels[i].tolist()
            for node_idx, link in enumerate(self.links):
                records.append(
                    {
                        "sample_index": i,
                        "timestamp": ts.isoformat(),
                        "link": link,
                        "label": labels[node_idx],
                    }
                )
        df = pd.DataFrame.from_records(records)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def _resample_frame(
    df: pd.DataFrame, freq_minutes: int, delay_threshold: Optional[float] = None, column_delay: str = "Atraso"
) -> pd.DataFrame:
    """
    Resamples the dataframe to a uniform frequency.
    -----------------------------------------------
    Preserves numeric columns such as column_delay and Num_Hops after resampling.

    Args:
        df: DataFrame with columns Timestamp, column_delay and optionally Num_Hops
        freq_minutes: Resampling frequency in minutes
        delay_threshold: Fixed threshold for high delay(ms). If None, will be calculated by percentile later.
        column_delay: Name of the delay column to use
    """
    frame = df.copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=False)
    frame = frame.sort_values("Timestamp")
    frame = frame.set_index("Timestamp")
    rule = f"{int(freq_minutes)}min"

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()

    agg = frame[numeric_cols].resample(rule).mean()

    if column_delay in agg.columns:
        agg[column_delay] = agg[column_delay].interpolate(
            method="linear", limit_direction="both"
        )
        agg[column_delay] = agg[column_delay].ffill().bfill()

    if "Num_Hops" in agg.columns:
        agg["Num_Hops"] = agg["Num_Hops"].interpolate(
            method="linear", limit_direction="both"
        )
        agg["Num_Hops"] = agg["Num_Hops"].ffill().bfill()

    if delay_threshold is not None and column_delay in agg.columns:
        agg["high_delay"] = (agg[column_delay] > delay_threshold).astype(int)
    agg = agg.dropna(subset=[column_delay], how="all")
    agg = agg.reset_index()
    return agg


def parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(description="Build DelayGraphDataset for GNN")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets_generated"))
    parser.add_argument("--links", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--min-corr", type=float, default=0.3)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--delay-threshold", type=float, default=None)
    parser.add_argument("--delay-percentile", type=float, default=85.0)
    parser.add_argument("--topology-weight", type=float, default=0.4)
    parser.add_argument("--column-delay", type=str, default="Atraso")
    parser.add_argument(
        "--output-joblib",
        type=Path,
        default=Path("results_artifacts/gnn_dataset.joblib"),
        help="Path to save the pre-built dataset (.joblib)",
    )
    parser.add_argument(
        "--generate-all-horizons",
        action="store_true",
        help="Generate datasets for all horizons (1, 3, 5, 10, 30 min)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_artifacts/datasets"),
        help="Output directory when generating multiple horizons",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.generate_all_horizons:
        horizons = [1, 3, 5, 10, 30]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print(f"Generating GNN datasets for {len(horizons)} horizons")
        print("=" * 60)
        
        for horizon in horizons:
            print(f"\n[{horizon}min] Building dataset...")
            dataset = DelayGraphDataset(
                data_dir=args.data_dir,
                links=args.links,
                window_size=args.window_size,
                horizon_minutes=horizon,
                min_corr=args.min_corr,
                limit_samples=args.limit_samples,
                delay_threshold=args.delay_threshold,
                delay_percentile=args.delay_percentile,
                topology_weight=args.topology_weight,
                column_delay=args.column_delay,
            )
            print(f"[{horizon}min] Built: {len(dataset)} samples, {dataset.num_node_features} features")
            
            output_path = args.output_dir / f"gnn_dataset_w{args.window_size}_h{horizon}min.joblib"
            dataset.save_joblib(output_path)
            print(f"[{horizon}min] Saved to {output_path}")
        
        print("\n" + "=" * 60)
        print("All GNN datasets generated successfully!")
        print(f"Location: {args.output_dir}")
        print("=" * 60)
    else:
        dataset = DelayGraphDataset(
            data_dir=args.data_dir,
            links=args.links,
            window_size=args.window_size,
            horizon_minutes=args.horizon_minutes,
            min_corr=args.min_corr,
            limit_samples=args.limit_samples,
            delay_threshold=args.delay_threshold,
            delay_percentile=args.delay_percentile,
            topology_weight=args.topology_weight,
            column_delay=args.column_delay,
        )

        print(f"[INFO] Built DelayGraphDataset with {len(dataset)} samples and {dataset.num_nodes} nodes")
        dataset.save_joblib(args.output_joblib)
        print(f"[INFO] Saved pre-built dataset to {args.output_joblib}")
