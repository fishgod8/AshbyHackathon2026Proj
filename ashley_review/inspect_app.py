"""Local Streamlit viewer for Ashley's reference/validation review results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = PROJECT_ROOT / "experiments/ashley_review/split_manifest.csv"
DEFAULT_SCORES = PROJECT_ROOT / "experiments/ashley_review/validation_scores.csv"
DEFAULT_GATE_CONFIG = PROJECT_ROOT / "experiments/ashley_review/reference_gate_config.json"
DEFAULT_METADATA = PROJECT_ROOT / "features/crc_val_musk/metadata.csv"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "features/crc_val_musk/embeddings.npy"
DEFAULT_EXTRACTION_CONFIG = PROJECT_ROOT / "features/crc_val_musk/extraction_config.json"


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def load_review_data(
    split_path: str | Path,
    scores_path: str | Path,
    gate_config_path: str | Path,
    metadata_path: str | Path,
    embeddings_path: str | Path,
    extraction_config_path: str | Path = DEFAULT_EXTRACTION_CONFIG,
) -> dict[str, Any]:
    """Load only reference/validation records and feature rows for the viewer."""
    split_rows = _read_dict_rows(Path(split_path))
    required_split_columns = {"image_id", "row_index", "label", "split"}
    if not split_rows or not required_split_columns.issubset(split_rows[0]):
        raise ValueError("split manifest is missing required columns")

    selected_rows: dict[str, dict[str, str]] = {}
    for position, row in enumerate(split_rows):
        if row["split"] not in {"reference", "validation", "test"}:
            raise ValueError(f"unexpected split value at manifest row {position}")
        if row["split"] == "test":
            continue
        if int(row["row_index"]) != position:
            raise ValueError(f"manifest row_index mismatch at row {position}")
        if row["image_id"] in selected_rows:
            raise ValueError(f"duplicate image_id in reference/validation manifest: {row['image_id']}")
        selected_rows[row["image_id"]] = row

    reference_rows = [row for row in selected_rows.values() if row["split"] == "reference"]
    validation_rows = [row for row in selected_rows.values() if row["split"] == "validation"]
    if not reference_rows or not validation_rows:
        raise ValueError("split manifest must contain reference and validation rows")

    with Path(gate_config_path).open(encoding="utf-8") as file_handle:
        gate_config = json.load(file_handle)
    if gate_config.get("reference_split") != "reference" or gate_config.get("query_split") != "validation":
        raise ValueError("gate config does not describe the reference/validation workflow")
    integrity = gate_config.get("integrity", {})
    manifest_digest = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
    if manifest_digest != integrity.get("split_manifest_sha256"):
        raise ValueError("split manifest SHA-256 does not match the saved gate config")
    extraction_config_path = Path(extraction_config_path)
    extraction_digest = hashlib.sha256(extraction_config_path.read_bytes()).hexdigest()
    if extraction_digest != integrity.get("extraction_config_sha256"):
        raise ValueError("extraction config SHA-256 does not match the saved gate config")
    with extraction_config_path.open(encoding="utf-8") as file_handle:
        extraction_config = json.load(file_handle)
    for key in (
        "model",
        "checkpoint",
        "checkpoint_sha256",
        "preprocessing",
        "extraction",
        "software_versions",
    ):
        if gate_config.get("feature_settings", {}).get(key) != extraction_config.get(key):
            raise ValueError(f"gate feature settings do not match extraction config field {key}")

    scores_rows = _read_dict_rows(Path(scores_path))
    if not scores_rows:
        raise ValueError("validation score file is empty")
    score_fields = set(scores_rows[0])
    required_score_fields = {"image_id", "label", "reference_distance", "insufficient_reference_support"}
    required_neighbor_fields = {
        field
        for rank in range(1, 6)
        for field in (f"neighbor_{rank}_image_id", f"neighbor_{rank}_distance")
    }
    if not required_score_fields.union(required_neighbor_fields).issubset(score_fields):
        raise ValueError("validation score file is missing expected columns")
    score_by_id = {row["image_id"]: row for row in scores_rows}
    validation_ids = {row["image_id"] for row in validation_rows}
    if set(score_by_id) != validation_ids:
        raise ValueError("validation score IDs do not exactly match the validation split")
    if len(score_by_id) != len(scores_rows):
        raise ValueError("validation score file contains duplicate image IDs")
    for row in validation_rows:
        score_row = score_by_id[row["image_id"]]
        if score_row["label"] != row["label"]:
            raise ValueError(f"validation label mismatch for {row['image_id']}")
        distance = float(score_row["reference_distance"])
        expected_flag = distance > float(gate_config["threshold"])
        if score_row["insufficient_reference_support"].strip().lower() not in {"true", "false"}:
            raise ValueError(f"invalid saved review flag for {row['image_id']}")
        if (score_row["insufficient_reference_support"].strip().lower() == "true") != expected_flag:
            raise ValueError(f"saved review flag does not match threshold for {row['image_id']}")
        neighbor_ids = [score_row[f"neighbor_{rank}_image_id"] for rank in range(1, 6)]
        if any(neighbor_id not in selected_rows or selected_rows[neighbor_id]["split"] != "reference" for neighbor_id in neighbor_ids):
            raise ValueError(f"non-reference neighbor found for validation image {row['image_id']}")
        neighbor_distances = [float(score_row[f"neighbor_{rank}_distance"]) for rank in range(1, 6)]
        if neighbor_distances != sorted(neighbor_distances) or neighbor_distances[-1] != distance:
            raise ValueError(f"saved neighbor distances are inconsistent for {row['image_id']}")

    needed_ids = set(selected_rows)
    metadata: dict[str, dict[str, str]] = {}
    with Path(metadata_path).open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if not {"image_id", "image_path", "label"}.issubset(reader.fieldnames or ()):
            raise ValueError("metadata is missing image_id,image_path,label columns")
        for row in reader:
            if row["image_id"] not in needed_ids:
                continue
            if row["image_id"] in metadata:
                raise ValueError(f"duplicate metadata image_id: {row['image_id']}")
            metadata[row["image_id"]] = row
    if set(metadata) != needed_ids:
        raise ValueError("metadata does not contain every reference and validation image")
    for image_id, row in selected_rows.items():
        if metadata[image_id]["label"] != row["label"]:
            raise ValueError(f"metadata label mismatch for {image_id}")

    feature_config = gate_config.get("feature_settings", {})
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    expected_shape = feature_config.get("shape")
    if list(embeddings.shape) != expected_shape:
        raise ValueError("embedding array shape does not match saved gate feature settings")
    if str(embeddings.dtype) != feature_config.get("dtype"):
        raise ValueError("embedding dtype does not match saved gate feature settings")

    reference_indices = np.asarray([int(row["row_index"]) for row in reference_rows], dtype=np.intp)
    validation_indices = np.asarray([int(row["row_index"]) for row in validation_rows], dtype=np.intp)
    reference_raw = np.asarray(embeddings[reference_indices])
    validation_raw = np.asarray(embeddings[validation_indices])
    if not np.isfinite(reference_raw).all() or not np.isfinite(validation_raw).all():
        raise ValueError("reference or validation embeddings contain non-finite values")
    reference_norms = np.linalg.norm(reference_raw.astype(np.float64), axis=1)
    validation_norms = np.linalg.norm(validation_raw.astype(np.float64), axis=1)
    if np.any(reference_norms == 0) or np.any(validation_norms == 0):
        raise ValueError("reference or validation embeddings contain zero vectors")
    reference_normalized = reference_raw.astype(np.float64) / reference_norms[:, None]
    validation_normalized = validation_raw.astype(np.float64) / validation_norms[:, None]
    used_features_digest = hashlib.sha256()
    used_features_digest.update(np.ascontiguousarray(reference_raw).tobytes())
    used_features_digest.update(np.ascontiguousarray(validation_raw).tobytes())
    if used_features_digest.hexdigest() != integrity.get(
        "sha256_of_only_used_reference_then_validation_embedding_rows"
    ):
        raise ValueError("reference/validation embedding fingerprint does not match the saved gate config")

    validation_data = []
    for row in validation_rows:
        score = score_by_id[row["image_id"]]
        validation_data.append(
            {
                "image_id": row["image_id"],
                "label": row["label"],
                "reference_distance": float(score["reference_distance"]),
                "needs_review": score["insufficient_reference_support"].strip().lower() == "true",
                "score": score,
                "image_path": metadata[row["image_id"]]["image_path"],
            }
        )

    return {
        "reference_rows": reference_rows,
        "validation_rows": validation_data,
        "metadata": metadata,
        "gate_config": gate_config,
        "reference_embeddings": reference_normalized,
        "validation_embeddings": validation_normalized,
        "reference_ids": [row["image_id"] for row in reference_rows],
        "validation_ids": [row["image_id"] for row in validation_rows],
    }


def fit_reference_pca(
    reference_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit three-component PCA on reference only and transform both allowed splits."""
    if reference_embeddings.ndim != 2 or validation_embeddings.ndim != 2:
        raise ValueError("PCA inputs must be two-dimensional arrays")
    if reference_embeddings.shape[1] != validation_embeddings.shape[1]:
        raise ValueError("reference and validation embedding dimensions differ")
    if reference_embeddings.shape[0] < 3:
        raise ValueError("at least three reference rows are required for 3D PCA")
    mean = reference_embeddings.mean(axis=0)
    centered_reference = reference_embeddings - mean
    _, singular_values, components = np.linalg.svd(centered_reference, full_matrices=False)
    first_components = components[:3]
    total_variance = float(np.sum(singular_values**2))
    if total_variance <= 0 or not np.isfinite(total_variance):
        raise ValueError("reference embeddings do not have finite PCA variance")
    cumulative_variance = float(np.sum(singular_values[:3] ** 2) / total_variance)
    reference_coordinates = centered_reference @ first_components.T
    validation_coordinates = (validation_embeddings - mean) @ first_components.T
    return reference_coordinates, validation_coordinates, cumulative_variance


def _resolve_image(dataset_root: Path, relative_path: str) -> Path:
    return dataset_root / Path(relative_path)


def main() -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as error:
        raise SystemExit(
            "Viewer dependencies are missing. Install only these packages into the selected environment:\n"
            "  python -m pip install streamlit plotly\n"
            "Then launch with the configured Python interpreter."
        ) from error

    st.set_page_config(page_title="Ashley Validation Review", layout="wide")
    st.title("验证集开发查看器")
    st.caption("仅查看个人 reference / validation 数据；这不是独立性能评估。")
    st.info("真实标签仅用于开发查看，不参与距离判定。距离越大，参考支持越弱；这些距离不是分类概率。")

    with st.sidebar:
        st.header("数据位置")
        try:
            with DEFAULT_GATE_CONFIG.open(encoding="utf-8") as handle:
                default_gate = json.load(handle)
            configured_root = default_gate.get("feature_settings", {}).get("dataset_root")
        except (OSError, json.JSONDecodeError):
            configured_root = None
        default_dataset_root = configured_root or str(PROJECT_ROOT / "CRC-VAL-HE-7K")
        dataset_root_text = st.text_input("数据集根目录", value=default_dataset_root)
        show_3d = st.checkbox("显示可选 3D PCA 地图", value=False)

    @st.cache_data(show_spinner="读取 reference 与 validation 数据…")
    def cached_load() -> dict[str, Any]:
        return load_review_data(
            DEFAULT_SPLIT,
            DEFAULT_SCORES,
            DEFAULT_GATE_CONFIG,
            DEFAULT_METADATA,
            DEFAULT_EMBEDDINGS,
            DEFAULT_EXTRACTION_CONFIG,
        )

    try:
        data = cached_load()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"无法读取个人复核数据：{error}")
        st.stop()

    rows = data["validation_rows"]
    threshold = float(data["gate_config"]["threshold"])
    flagged_count = sum(row["needs_review"] for row in rows)
    total_count = len(rows)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Reference 数量", f"{len(data['reference_rows']):,}")
    summary_columns[1].metric("Validation 数量", f"{total_count:,}")
    summary_columns[2].metric("固定距离阈值", f"{threshold:.6f}")
    summary_columns[3].metric(
        "被标记", f"{flagged_count:,} ({flagged_count / total_count:.2%})"
    )

    st.subheader("Validation 距离分布")
    distances = np.asarray([row["reference_distance"] for row in rows], dtype=float)
    counts, edges = np.histogram(distances, bins=50)
    figure = go.Figure(
        data=[go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges))]
    )
    figure.add_vline(x=threshold, line_color="#c24635", line_width=2, line_dash="dash")
    figure.add_annotation(
        x=threshold,
        y=1,
        yref="paper",
        text=f"固定阈值 {threshold:.6f}",
        showarrow=True,
        arrowhead=2,
        xanchor="left",
    )
    figure.update_layout(
        xaxis_title="到第 5 个最近 reference 邻居的 cosine distance",
        yaxis_title="Validation 图片数",
        bargap=0.04,
        height=420,
        margin={"l": 30, "r": 30, "t": 45, "b": 35},
    )
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Validation 图片")
    filter_columns = st.columns(3)
    flagged_only = filter_columns[0].checkbox("只看被标记图片", value=False)
    labels = sorted({row["label"] for row in rows})
    selected_label = filter_columns[1].selectbox("类别筛选", ["全部", *labels])
    sort_descending = filter_columns[2].checkbox("按距离从大到小排序", value=True)
    visible_rows = [
        row
        for row in rows
        if (not flagged_only or row["needs_review"])
        and (selected_label == "全部" or row["label"] == selected_label)
    ]
    visible_rows.sort(key=lambda row: row["reference_distance"], reverse=sort_descending)
    display_rows = [
        {
            "image_id": row["image_id"],
            "真实组织标签": row["label"],
            "reference_distance": row["reference_distance"],
            "needs_review": row["needs_review"],
        }
        for row in visible_rows
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    if not visible_rows:
        st.warning("当前筛选没有图片。")
        st.stop()

    visible_ids = [row["image_id"] for row in visible_rows]
    default_id = visible_ids[0]
    selected_id = st.selectbox(
        "选择一张 validation 图片查看详情",
        visible_ids,
        index=0,
        format_func=lambda image_id: next(
            f"{image_id} · {next(row['reference_distance'] for row in rows if row['image_id'] == image_id):.6f}"
            for row in rows
            if row["image_id"] == image_id
        ),
        key="validation_detail_image",
    )
    selected = next(row for row in rows if row["image_id"] == selected_id)
    dataset_root = Path(dataset_root_text).expanduser()

    st.subheader("单张复核详情")
    image_column, detail_column = st.columns([1, 1])
    original_path = _resolve_image(dataset_root, selected["image_path"])
    with image_column:
        st.markdown(f"**Validation 图片：** `{selected['image_id']}`")
        if original_path.is_file():
            st.image(str(original_path), caption=selected["image_path"], use_container_width=True)
        else:
            st.warning(f"原图缺失：未找到 `{original_path}`。请检查数据集根目录配置。")
    with detail_column:
        st.write(f"真实组织标签（仅供开发查看）：**{selected['label']}**")
        st.write(f"第 5 个邻居距离：**{selected['reference_distance']:.8f}**")
        st.write(f"固定阈值：**{threshold:.8f}**")
        st.write(f"触发理由：**LOW_REFERENCE_SUPPORT**" if selected["needs_review"] else "触发理由：无")
        st.caption("距离提醒不是分类结果，不代表发现未知疾病、训练数据缺少某种疾病或分类错误。")

    st.markdown("**五个 reference 邻居**")
    neighbor_columns = st.columns(5)
    selected_score = selected["score"]
    for rank, column in enumerate(neighbor_columns, start=1):
        neighbor_id = selected_score[f"neighbor_{rank}_image_id"]
        neighbor_meta = data["metadata"][neighbor_id]
        neighbor_path = _resolve_image(dataset_root, neighbor_meta["image_path"])
        with column:
            st.caption(
                f"{rank}. {neighbor_id}\n标签：{neighbor_meta['label']}\n"
                f"Cosine distance：{float(selected_score[f'neighbor_{rank}_distance']):.8f}"
            )
            if neighbor_path.is_file():
                st.image(str(neighbor_path), use_container_width=True)
            else:
                st.warning(f"Reference 原图缺失：`{neighbor_path}`")

    if show_3d:
        st.subheader("Reference PCA 3D 地图")
        st.warning("投影会丢失信息，图中距离不等于原特征距离。PCA 只用于可视化；提醒仍使用原始 1024 维空间中的固定 reference 距离规则。")

        @st.cache_data(show_spinner="仅在 reference embeddings 上拟合 PCA…")
        def cached_pca(reference_embeddings: np.ndarray, validation_embeddings: np.ndarray):
            return fit_reference_pca(reference_embeddings, validation_embeddings)

        try:
            reference_xyz, validation_xyz, explained = cached_pca(
                data["reference_embeddings"], data["validation_embeddings"]
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            st.error(f"无法生成 reference PCA 投影：{error}")
        else:
            st.caption(f"前三个主成分累计解释方差：{explained:.2%}")
            reference_positions = {image_id: i for i, image_id in enumerate(data["reference_ids"])}
            validation_positions = {image_id: i for i, image_id in enumerate(data["validation_ids"])}
            palette = ["#3977a8", "#c05746", "#4d8a69", "#b07a2a", "#8167a9", "#438b91", "#b6537a", "#777c83", "#8b8a48"]
            label_color = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
            map_figure = go.Figure()
            for label in labels:
                positions = [
                    i for i, row in enumerate(data["reference_rows"]) if row["label"] == label
                ]
                map_figure.add_trace(
                    go.Scatter3d(
                        x=reference_xyz[positions, 0],
                        y=reference_xyz[positions, 1],
                        z=reference_xyz[positions, 2],
                        mode="markers",
                        name=f"Reference · {label}",
                        marker={"size": 2.5, "opacity": 0.48, "color": label_color[label]},
                        text=[data["reference_ids"][i] for i in positions],
                        hovertemplate="%{text}<extra>Reference " + label + "</extra>",
                    )
                )
            query_xyz = validation_xyz[validation_positions[selected_id]]
            selected_score = selected["score"]
            neighbor_ids = [selected_score[f"neighbor_{rank}_image_id"] for rank in range(1, 6)]
            neighbor_positions = [reference_positions[neighbor_id] for neighbor_id in neighbor_ids]
            neighbor_xyz = reference_xyz[neighbor_positions]
            for neighbor_point in neighbor_xyz:
                map_figure.add_trace(
                    go.Scatter3d(
                        x=[query_xyz[0], neighbor_point[0]],
                        y=[query_xyz[1], neighbor_point[1]],
                        z=[query_xyz[2], neighbor_point[2]],
                        mode="lines",
                        line={"color": "#54616a", "width": 2, "dash": "dash"},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            map_figure.add_trace(
                go.Scatter3d(
                    x=neighbor_xyz[:, 0], y=neighbor_xyz[:, 1], z=neighbor_xyz[:, 2],
                    mode="markers", name="当前图片的 5 个 reference 邻居",
                    marker={"size": 7, "color": "#f0a43a", "symbol": "diamond", "line": {"color": "#5c4523", "width": 1}},
                    text=neighbor_ids,
                    hovertemplate="%{text}<extra>Reference neighbor</extra>",
                )
            )
            map_figure.add_trace(
                go.Scatter3d(
                    x=[query_xyz[0]], y=[query_xyz[1]], z=[query_xyz[2]],
                    mode="markers", name="所选 validation 图片",
                    marker={"size": 9, "color": "#e04444", "symbol": "cross", "line": {"color": "#641f1f", "width": 2}},
                    text=[selected_id],
                    hovertemplate="%{text}<extra>Validation</extra>",
                )
            )
            map_figure.update_layout(
                scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
                height=680,
                margin={"l": 0, "r": 0, "t": 20, "b": 0},
                legend={"orientation": "h"},
            )
            st.plotly_chart(map_figure, use_container_width=True)


if __name__ == "__main__":
    main()