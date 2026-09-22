#!/usr/bin/env python3
"""MR-RATE-1K四标签AMEF-MIL五折；复用CT-RATE训练/评价协议。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

from prepare_mrrate_1k_experiment import (
    OUT as INPUT,
    LABELS,
    ROOT,
    save_json,
    sha256,
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)

import run_cq500_table2_multimodal_baselines as fusion_base

from run_physionet_ct_ich_table2_baselines import (
    CachedFeatureBackbone,
    collate_bags,
)


MODEL_KEY = "amef_multimodal"

MODELS = {
    MODEL_KEY:
        "AMEF-MIL",
}

SETTINGS = {
    **fusion_base.SETTINGS,

    "epochs":
        30,

    "batch_size":
        16,

    "learning_rate":
        2e-4,

    "weight_decay":
        0.02,

    "warmup_ratio":
        0.2,

    "max_instances":
        64,

    "instance_dropout":
        0.25,

    "base_seed":
        42,

    "text_max_length":
        512,

    "text_vocab_size":
        8192,
}

SOURCE_POSITIONS = []
SOURCE_COUNTS = []


def read_inputs():
    rows = json.loads(
        (
            INPUT
            / "samples.json"
        ).read_text()
    )

    folds = json.loads(
        (
            INPUT
            / "patient_folds.json"
        ).read_text()
    )

    assert len(rows) == 1000

    assert len({
        row["patient_uid"]
        for row in rows
    }) == 1000

    assert [
        row["case_index"]
        for row in rows
    ] == list(range(1000))

    assert folds["labels"] == LABELS

    assert sorted(
        i
        for group in folds["folds"]
        for i in group
    ) == list(range(1000))

    assert [
        len(group)
        for group in folds["folds"]
    ] == [200] * 5

    y = np.asarray(
        [
            row["labels"]
            for row in rows
        ],
        dtype=np.int64,
    )

    assert y.sum(0).tolist() == [
        379,
        320,
        270,
        231,
    ]

    assert [
        int(
            (
                y.sum(1) == n
            ).sum()
        )
        for n in range(5)
    ] == [
        350,
        292,
        191,
        142,
        25,
    ]

    return rows, folds, y


def model_parameters():
    import yaml

    from scripts.task3_apro_cope_ablation_scheduler import (
        base_model_params,
    )

    params = base_model_params(
        "apro_full"
    )

    main = yaml.safe_load(
        (
            ROOT
            / "src/configs/task3/t3_main_model.yaml"
        ).read_text()
    )

    params[
        "label_query_consistency_weight"
    ] = float(
        main[
            "model"
        ][
            "params"
        ][
            "label_query_consistency_weight"
        ]
    )

    params[
        "image_aux_weight"
    ] = 0.0

    return params


def protocol():
    tracked = [
        Path(__file__),

        ROOT
        / "src/scripts/prepare_mrrate_1k_experiment.py",

        ROOT
        / "src/scripts/run_cq500_table2_multimodal_baselines.py",

        ROOT
        / "src/scripts/run_physionet_ct_ich_table2_baselines.py",

        ROOT
        / "src/scripts/task3_apro_cope_ablation_scheduler.py",

        ROOT
        / "src/exp_8/models.py",

        ROOT
        / "src/sotas/task2/multimodal_sotas.py",

        ROOT
        / "src/training/losses.py",

        ROOT
        / "src/configs/task3/t3_main_model.yaml",

        INPUT
        / "samples.json",

        INPUT
        / "patient_folds.json",

        INPUT
        / "preparation_protocol.json",
    ]

    value = {
        "dataset":
            "MR-RATE-1K",

        "cases":
            1000,

        "labels":
            LABELS,

        "model":
            MODELS,

        "settings":
            SETTINGS,

        "model_parameters":
            model_parameters(),

        "source_sha256": {
            str(path):
                sha256(path)
            for path in tracked
        },

        "split":
            "固定五折；"
            "test=k，validation=(k+1)%5，"
            "其余三折训练；600/200/200",

        "image":
            "全部MRI series组成检查级有序序列；"
            "最多64实例；冻结ConvNeXt-Tiny特征",

        "text":
            "仅findings；MASKTARGET统一掩码；"
            "不输入impression/clinical_information/technique",

        "amef":
            "APro-CoPE + label-wise attention + "
            "label hypergraph + TextCNN + "
            "label-query cross-attention + gated fusion",

        "position":
            "使用实例在完整展平MRI序列中的global index"
            "以及original instance count",

        "training":
            "30 epochs, batch16, AdamW lr2e-4, "
            "wd0.02, warmup20%, cosine, "
            "ASL + 0.01 label-query loss, "
            "25% instance dropout",

        "precision":
            "AMEF forward FP32",

        "selection":
            "最低validation classification loss选权重；"
            "validation逐标签调阈值；test不参与选择",

        "primary_metric":
            "五折test Macro-F1 mean±SD；"
            "另报OOF及共阳性子集",
    }

    value[
        "protocol_sha256"
    ] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    return value


def load_features(rows):
    global SOURCE_POSITIONS
    global SOURCE_COUNTS

    protocol = json.loads(
        (
            INPUT
            / "preparation_protocol.json"
        ).read_text()
    )

    digest = protocol[
        "preparation_sha256"
    ]

    bags = []
    SOURCE_POSITIONS = []
    SOURCE_COUNTS = []

    for row in tqdm(
        rows,
        desc="读取MR-RATE-1K特征",
    ):
        path = (
            INPUT
            / "features"
            / f"{row['case_index']:04d}.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"缺少特征：{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as cache:

            assert (
                str(
                    cache["study_uid"]
                )
                ==
                row["study_uid"]
            )

            assert (
                str(
                    cache[
                        "preparation_sha256"
                    ]
                )
                ==
                digest
            )

            features = (
                cache["features"]
                .astype(np.float32)
            )

            positions = (
                cache["slice_indices"]
                .astype(np.int64)
            )

            count = int(
                cache[
                    "original_count"
                ]
            )

            assert (
                features.shape
                ==
                (
                    len(positions),
                    768,
                )
            )

            assert (
                1
                <= len(positions)
                <= 64
            )

            assert np.isfinite(
                features
            ).all()

            assert np.all(
                np.diff(
                    positions
                ) > 0
            )

            assert (
                0
                <= positions[0]
                <= positions[-1]
                < count
            )

            bags.append(
                features
            )

            SOURCE_POSITIONS.append(
                positions
            )

            SOURCE_COUNTS.append(
                count
            )

    return bags


def build_model(
    key,
    parameters,
):
    import torch

    from exp_8.models import (
        Exp12AProCoPEWatchCrossAttentionTextCNNModel,
    )

    assert key == MODEL_KEY

    params = dict(
        parameters
    )

    auxiliary = {
        name.removesuffix(
            "_weight"
        ):
            params.pop(name)

        for name
        in list(params)

        if name.endswith(
            "_weight"
        )
    }

    model = (
        Exp12AProCoPEWatchCrossAttentionTextCNNModel(
            **params,
            pretrained=False,
            num_labels=
                len(LABELS),
        )
    )

    model.instance_encoder.backbone = (
        CachedFeatureBackbone(
            input_dim=768,
            output_dim=
                params[
                    "feature_dim"
                ],
            dropout=
                params[
                    "dropout"
                ],
        )
    )

    torch.backends.mha.set_fastpath_enabled(
        False
    )

    return model, auxiliary


def loader(
    bags,
    labels,
    indices,
    training,
    seed,
):
    import torch

    from torch.utils.data import (
        DataLoader,
        Dataset,
    )

    class OrderedFeatureBags(
        Dataset
    ):
        def __len__(self):
            return len(indices)

        def __getitem__(
            self,
            item,
        ):
            case = int(
                indices[item]
            )

            bag = bags[case]

            selected = np.arange(
                len(bag),
                dtype=np.int64,
            )

            if (
                training
                and SETTINGS[
                    "instance_dropout"
                ] > 0
                and len(selected) > 1
            ):
                keep = max(
                    1,
                    int(
                        round(
                            len(selected)
                            *
                            (
                                1.0
                                -
                                SETTINGS[
                                    "instance_dropout"
                                ]
                            )
                        )
                    ),
                )

                chosen = np.sort(
                    np.random.choice(
                        len(selected),
                        keep,
                        replace=False,
                    )
                )

                selected = (
                    selected[
                        chosen
                    ]
                )

            return (
                torch.from_numpy(
                    bag[selected]
                ),

                torch.tensor(
                    labels[case],
                    dtype=torch.float32,
                ),

                case,

                torch.from_numpy(
                    selected
                ),

                SOURCE_COUNTS[
                    case
                ],
            )

    def collate(records):
        (
            features,
            mask,
            targets,
            case_ids,
        ) = collate_bags(
            [
                record[:3]
                for record
                in records
            ]
        )

        cached_positions = (
            torch.full(
                mask.shape,
                -1,
                dtype=torch.long,
            )
        )

        for j, record in enumerate(
            records
        ):
            cached_positions[
                j,
                :len(record[3])
            ] = record[3]

        counts = torch.tensor(
            [
                record[4]
                for record
                in records
            ],
            dtype=torch.long,
        )

        return (
            features,
            mask,
            targets,
            case_ids,
            cached_positions,
            counts,
        )

    return DataLoader(
        OrderedFeatureBags(),
        batch_size=
            SETTINGS[
                "batch_size"
            ],
        shuffle=training,
        num_workers=0,
        collate_fn=collate,
        generator=
            torch.Generator()
            .manual_seed(seed),
    )


def forward_batch(
    model,
    batch,
    text_ids,
    text_mask,
    device,
    training=False,
):
    import torch

    (
        features,
        mask,
        labels,
        case_ids,
        cached_positions,
        counts,
    ) = batch

    absolute_positions = (
        torch.full_like(
            cached_positions,
            -1,
        )
    )

    for j, case in enumerate(
        case_ids.tolist()
    ):
        valid = (
            cached_positions[j]
            >= 0
        )

        local = (
            cached_positions[
                j,
                valid
            ]
            .numpy()
        )

        absolute_positions[
            j,
            valid
        ] = torch.from_numpy(
            SOURCE_POSITIONS[
                case
            ][local]
        )

    kwargs = {
        "images":
            features.to(
                device
            ),

        "mask":
            mask.to(
                device
            ),

        "watch_token_ids":
            text_ids[
                case_ids
            ].to(device),

        "watch_token_mask":
            text_mask[
                case_ids
            ].to(device),

        "instance_indices":
            absolute_positions.to(
                device
            ),

        "original_image_counts":
            counts.to(
                device
            ),
    }

    if training:
        kwargs["labels"] = (
            labels.to(
                device
            )
        )

    with torch.autocast(
        device_type=
            torch.device(
                device
            ).type,
        enabled=False,
    ):
        return model(
            **kwargs
        )


def configure_base():
    fusion_base.LABEL_NAMES = (
        LABELS
    )

    fusion_base.MODELS = (
        MODELS
    )

    fusion_base.SETTINGS = (
        SETTINGS
    )

    fusion_base.build_model = (
        build_model
    )

    fusion_base.loader = (
        loader
    )

    fusion_base.forward_batch = (
        forward_batch
    )


def worker(args):
    import torch

    torch.set_num_threads(4)

    configure_base()

    current = protocol()

    saved = json.loads(
        (
            args.output_dir
            / "protocol.json"
        ).read_text()
    )

    assert (
        saved[
            "protocol_sha256"
        ]
        ==
        current[
            "protocol_sha256"
        ]
    )

    rows, folds, y = (
        read_inputs()
    )

    bags = load_features(
        rows
    )

    ids = np.arange(
        1000,
        dtype=np.int64,
    )

    texts = {
        row["case_index"]:
            row[
                "findings_masked"
            ]

        for row
        in rows
    }

    text_ids, text_mask = (
        fusion_base
        .encode_descriptions(
            texts,
            ids,
        )
    )

    params = (
        model_parameters()
    )

    jobs = [
        (
            MODEL_KEY,
            fold,
        )
        for fold
        in range(5)
    ]

    for index, (
        key,
        fold,
    ) in enumerate(jobs):

        if (
            index
            % len(
                args.devices
            )
            !=
            args.worker_index
        ):
            continue

        marker = (
            args.output_dir
            / f"fold_{fold + 1}"
            / key
            / "completed.json"
        )

        if marker.exists():
            complete = json.loads(
                marker.read_text()
            )

            assert (
                complete[
                    "protocol_sha256"
                ]
                ==
                current[
                    "protocol_sha256"
                ]
            )

            continue

        try:
            fusion_base.train_fold(
                args,
                key,
                fold,
                ids,
                bags,
                y,
                folds,
                text_ids,
                text_mask,
                params,
                current[
                    "protocol_sha256"
                ],
            )

            gc.collect()
            torch.cuda.empty_cache()

        except Exception:
            save_json(
                args.output_dir
                / "errors"
                / f"fold_{fold + 1}.json",

                {
                    "traceback":
                        traceback
                        .format_exc()
                },
            )

            raise


def aggregate(
    args,
    digest,
):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    rows, _, targets = (
        read_inputs()
    )

    metrics = []
    predictions = []

    for fold in range(
        1,
        6,
    ):
        folder = (
            args.output_dir
            / f"fold_{fold}"
            / MODEL_KEY
        )

        if not (
            folder
            / "completed.json"
        ).exists():
            continue

        metric = json.loads(
            (
                folder
                / "test_metrics.json"
            ).read_text()
        )

        accepted_protocols = getattr(args, "accepted_protocols", {digest})
        assert metric["protocol_sha256"] in accepted_protocols

        metrics.append(
            metric
        )

        with (
            folder
            / "test_predictions.csv"
        ).open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:

            predictions.extend(
                {
                    **row,
                    "fold":
                        fold,
                }

                for row
                in csv.DictReader(
                    stream
                )
            )

    completed = len(metrics)

    save_json(
        args.output_dir
        / "progress.json",

        {
            "completed_fold_jobs":
                completed,

            "expected_fold_jobs":
                5,
        },
    )

    if completed != 5:
        return completed

    predictions.sort(
        key=lambda row:
            int(
                row["patient_id"]
            )
    )

    assert [
        int(
            row["patient_id"]
        )
        for row
        in predictions
    ] == list(range(1000))

    y = np.asarray([
        [
            int(
                row[
                    f"true_{name}"
                ]
            )
            for name
            in LABELS
        ]
        for row
        in predictions
    ])

    probabilities = np.asarray([
        [
            float(
                row[
                    f"prob_{name}"
                ]
            )
            for name
            in LABELS
        ]
        for row
        in predictions
    ])

    predictions_binary = (
        np.asarray([
            [
                int(
                    row[
                        f"pred_{name}"
                    ]
                )
                for name
                in LABELS
            ]
            for row
            in predictions
        ])
    )

    assert np.array_equal(
        y,
        targets,
    )

    assert np.isfinite(
        probabilities
    ).all()

    copositive = (
        y.sum(1)
        >= 2
    )

    summary = {
        "model_key":
            MODEL_KEY,

        "model":
            MODELS[
                MODEL_KEY
            ],

        "completed_folds":
            5,

        "oof_cases":
            1000,

        "macro_f1_mean":
            statistics.mean(
                metric[
                    "macro_f1"
                ]
                for metric
                in metrics
            ),

        "macro_f1_std":
            statistics.stdev(
                metric[
                    "macro_f1"
                ]
                for metric
                in metrics
            ),

        "macro_f1_fixed_0_5_mean":
            statistics.mean(
                metric[
                    "macro_f1_fixed_0_5"
                ]
                for metric
                in metrics
            ),

        "macro_f1_fixed_0_5_std":
            statistics.stdev(
                metric[
                    "macro_f1_fixed_0_5"
                ]
                for metric
                in metrics
            ),

        "oof_macro_f1":
            float(
                f1_score(
                    y,
                    predictions_binary,
                    average="macro",
                    zero_division=0,
                )
            ),

        "oof_micro_f1":
            float(
                f1_score(
                    y,
                    predictions_binary,
                    average="micro",
                    zero_division=0,
                )
            ),

        "oof_exact_match":
            float(
                accuracy_score(
                    y,
                    predictions_binary,
                )
            ),

        "oof_per_label_f1":
            dict(
                zip(
                    LABELS,
                    f1_score(
                        y,
                        predictions_binary,
                        average=None,
                        zero_division=0,
                    ).tolist(),
                )
            ),

        "oof_macro_auroc":
            float(
                roc_auc_score(
                    y,
                    probabilities,
                    average="macro",
                )
            ),

        "copositive_cases":
            int(
                copositive.sum()
            ),

        "copositive_macro_f1":
            float(
                f1_score(
                    y[copositive],
                    predictions_binary[
                        copositive
                    ],
                    average="macro",
                    zero_division=0,
                )
            ),

        "copositive_exact_match":
            float(
                accuracy_score(
                    y[copositive],
                    predictions_binary[
                        copositive
                    ],
                )
            ),

        "folds":
            metrics,
    }

    save_json(
        args.output_dir
        / "summary.json",

        [
            summary
        ],
    )

    for row in predictions:
        index = int(
            row["patient_id"]
        )

        row[
            "source_patient_uid"
        ] = rows[index][
            "patient_uid"
        ]

        row[
            "source_study_uid"
        ] = rows[index][
            "study_uid"
        ]

    with (
        args.output_dir
        / "amef_multimodal_oof_predictions.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=
                list(
                    predictions[0]
                ),
        )

        writer.writeheader()
        writer.writerows(
            predictions
        )

    report = (
        "# MR-RATE-1K AMEF-MIL 四标签五折\n\n"

        "1000名不同患者，每例一个MRI study；"
        "使用全部MRI series构成的检查级有序64实例；"
        "文本仅为直接目标词统一掩码后的findings。\n\n"

        f"- Macro-F1："
        f"{summary['macro_f1_mean']:.4f} ± "
        f"{summary['macro_f1_std']:.4f}\n"

        f"- 固定0.5 Macro-F1："
        f"{summary['macro_f1_fixed_0_5_mean']:.4f} ± "
        f"{summary['macro_f1_fixed_0_5_std']:.4f}\n"

        f"- OOF Macro-F1："
        f"{summary['oof_macro_f1']:.4f}\n"

        f"- OOF Micro-F1："
        f"{summary['oof_micro_f1']:.4f}\n"

        f"- OOF Macro-AUROC："
        f"{summary['oof_macro_auroc']:.4f}\n"

        f"- 共阳性病例："
        f"{summary['copositive_cases']}；"
        f"Macro-F1="
        f"{summary['copositive_macro_f1']:.4f}\n"
    )

    (
        args.output_dir
        / "results.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    return completed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=
            ROOT
            / "outputs/mr_rate_1k/amef_fivefold",
    )

    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[
            0,
            1,
            2,
            3,
        ],
    )

    parser.add_argument(
        "--worker-index",
        type=int,
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.worker_index
        is not None
    ):
        worker(args)
        return

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_base()

    with (
        args.output_dir
        / "run.lock"
    ).open("a") as lock:

        fcntl.flock(
            lock,
            fcntl.LOCK_EX
            | fcntl.LOCK_NB,
        )

        read_inputs()

        current = protocol()

        protocol_path = (
            args.output_dir
            / "protocol.json"
        )

        if protocol_path.exists():
            previous = json.loads(
                protocol_path
                .read_text()
            )

            if (
                previous.get(
                    "protocol_sha256"
                )
                !=
                current[
                    "protocol_sha256"
                ]
            ):
                raise ValueError(
                    "MR-RATE正式协议或源码变化；"
                    "禁止与旧结果混用"
                )

        save_json(
            protocol_path,
            current,
        )

        if args.audit_only:
            feature_count = len(
                list(
                    (
                        INPUT
                        / "features"
                    ).glob(
                        "*.npz"
                    )
                )
            )

            print(
                "协议核对通过："
                "1000例 / 4标签 / 5折；"
                f"当前特征 {feature_count}/1000。",
                flush=True,
            )

            return

        handles = []
        workers = []
        started = time.time()

        for index, device in enumerate(
            args.devices
        ):
            env = os.environ.copy()

            env.update(
                CUDA_VISIBLE_DEVICES=
                    str(device),

                OMP_NUM_THREADS=
                    "4",

                TOKENIZERS_PARALLELISM=
                    "false",
            )

            command = [
                sys.executable,
                "-u",
                str(
                    Path(__file__)
                    .resolve()
                ),

                "--worker-index",
                str(index),

                "--devices",
                *map(
                    str,
                    args.devices,
                ),

                "--output-dir",
                str(
                    args.output_dir
                ),
            ]

            handle = (
                args.output_dir
                / f"worker_{index}.log"
            ).open("a")

            handles.append(
                handle
            )

            workers.append(
                subprocess.Popen(
                    command,
                    env=env,
                    stdout=handle,
                    stderr=
                        subprocess.STDOUT,
                )
            )

        state = {
            "status":
                "running",

            "worker_pids":
                [
                    p.pid
                    for p in workers
                ],

            "started_unix":
                started,
        }

        save_json(
            args.output_dir
            / "run_state.json",
            state,
        )

        while any(
            p.poll() is None
            for p in workers
        ):
            completed = aggregate(
                args,
                current[
                    "protocol_sha256"
                ],
            )

            print(
                "AMEF-MIL已完成 "
                f"{completed}/5 折",
                flush=True,
            )

            time.sleep(15)

        completed = aggregate(
            args,
            current[
                "protocol_sha256"
            ],
        )

        codes = [
            p.returncode
            for p in workers
        ]

        state.update(
            status=(
                "complete"
                if (
                    completed == 5
                    and not any(codes)
                )
                else "incomplete"
            ),

            completed_fold_jobs=
                completed,

            worker_exit_codes=
                codes,

            finished_unix=
                time.time(),

            wall_seconds=
                (
                    time.time()
                    - started
                ),
        )

        save_json(
            args.output_dir
            / "run_state.json",
            state,
        )

        for handle in handles:
            handle.close()

        print(
            "MR-RATE-1K AMEF-MIL"
            "五折训练结束："
            f"{state['status']}",
            flush=True,
        )

        if (
            state["status"]
            != "complete"
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
