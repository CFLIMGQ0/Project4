#!/usr/bin/env python3
"""后台启动CT-RATE特征准备与文本训练，随后补齐全部105个模型折次。"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_ctrate_680_experiment import OUT, ROOT, save_json


def main():
    output = ROOT / "outputs/ct_rate_680"
    scripts = ROOT / "src/scripts"
    with (output / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # 先固定输入；特征CPU解压与文本GPU训练并行，随后训练全部图像/融合模型。
        subprocess.run([sys.executable, "-u", str(scripts / "prepare_ctrate_680_experiment.py"), "--tables-only"], check=True)
        text_models = ["hashed_mean_encoder", "vocab_attention_encoder", "textcnn_encoder", "bigru_encoder", "transformer_encoder"]
        handles = [(output / name).open("a") for name in ("feature_preparation.log", "text_training.log", "all_models_training.log")]
        state = {"status": "preparing_features_and_training_text", "pid": os.getpid(), "started_unix": time.time()}
        feature = subprocess.Popen([sys.executable, "-u", str(scripts / "prepare_ctrate_680_experiment.py")],
                                   stdout=handles[0], stderr=subprocess.STDOUT)
        text = subprocess.Popen([sys.executable, "-u", str(scripts / "run_ctrate_680_all_models.py"),
                                 "--devices", "0", "1", "--models", *text_models],
                                stdout=handles[1], stderr=subprocess.STDOUT)
        state.update(feature_pid=feature.pid, text_pid=text.pid)
        save_json(output / "pipeline_state.json", state)
        while feature.poll() is None or text.poll() is None:
            state.update(feature_exit_code=feature.poll(), text_exit_code=text.poll(), updated_unix=time.time())
            save_json(output / "pipeline_state.json", state)
            time.sleep(15)
        if feature.returncode or text.returncode:
            state.update(status="incomplete", finished_unix=time.time())
            save_json(output / "pipeline_state.json", state)
            raise SystemExit(1)
        feature_state = json.loads((OUT / "feature_state.json").read_text())
        if feature_state["status"] != "complete" or feature_state["completed_cases"] != 680:
            raise RuntimeError("特征尚未完整，停止训练图像模型")
        state["status"] = "training_all_models"
        process = subprocess.Popen([sys.executable, "-u", str(scripts / "run_ctrate_680_all_models.py")],
                                   stdout=handles[2], stderr=subprocess.STDOUT)
        state["training_pid"] = process.pid
        save_json(output / "pipeline_state.json", state)
        code = process.wait()
        state.update(status="complete" if code == 0 else "incomplete", training_exit_code=code, finished_unix=time.time())
        save_json(output / "pipeline_state.json", state)
        for handle in handles:
            handle.close()
        raise SystemExit(code)


if __name__ == "__main__":
    main()
