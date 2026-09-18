#!/usr/bin/env python3
"""用隔离的V2rayA实例解析用户提供的订阅文件并测试；不改正在使用的代理。"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
from urllib.parse import unquote, urlsplit

import requests


def safe_message(value):
    value = re.sub(r"[a-zA-Z0-9+.-]+://\S+", "[地址已隐藏]", str(value))
    value = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", "[凭据已隐藏]", value)
    return value[:400]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attachment", type=Path)
    args = parser.parse_args()
    encoded = re.sub(r"\s+", "", args.attachment.read_text())
    decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True).decode()
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    lines = [line for line in lines if not unquote(urlsplit(line).fragment).startswith(("剩余流量", "套餐到期"))]
    assert len(lines) == 54 and all(urlsplit(line).scheme in ("vless", "hysteria2") for line in lines)
    for port in (2217, 21170, 21171, 21172, 52354):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    work = Path(tempfile.mkdtemp(prefix="offline-import-20260914-", dir="/home/Lim/.config/v2raya"))
    config = work / "staging"
    config.mkdir(mode=0o700)
    print("私有暂存目录：", work, flush=True)
    (work / "source.txt").write_text("\n".join(lines))
    (work / "source.txt").chmod(0o600)
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy") and not key.startswith("V2RAYA_")}
    env["NO_PROXY"] = "*"
    env.update(V2RAYA_ADDRESS="127.0.0.1:2217", V2RAYA_CONFIG=str(config),
               V2RAYA_V2RAY_BIN="/usr/local/bin/v2raya_core", V2RAYA_CORE_TYPE="xray",
               V2RAYA_V2RAY_ASSETSDIR="/usr/local/share/v2raya", V2RAYA_PASSCHECKROOT="true",
               V2RAYA_LOG_LEVEL="warn", V2RAYA_IPV6_SUPPORT="off")
    log = (work / "staging.log").open("w")
    process = subprocess.Popen([
        "/usr/local/bin/v2raya", "--lite",
    ], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    session = requests.Session()
    session.trust_env = False
    root = "http://127.0.0.1:2217/api/"

    def api(method, endpoint, **kwargs):
        response = session.request(method, root + endpoint, timeout=90, **kwargs)
        data = response.json()
        if not response.ok or data.get("code") != "SUCCESS":
            raise RuntimeError(endpoint + ": " + safe_message(data.get("message", response.status_code)))
        return data.get("data")

    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("隔离V2rayA启动失败，请检查私有日志")
            try:
                api("GET", "version")
                break
            except requests.ConnectionError:
                time.sleep(0.3)
        else:
            raise RuntimeError("隔离V2rayA启动超时")
        token = api("POST", "account", json={"username": "temporary_import", "password": secrets.token_urlsafe(18)})["token"]
        session.headers["Authorization"] = token
        api("PUT", "ports", json={"socks5": 21170, "http": 21171, "socks5WithPac": 0,
                                    "httpWithPac": 21172, "vmess": 0, "api": {"port": 0, "services": None}})
        settings = api("GET", "setting")["setting"]
        settings.update(transparent="close", portSharing=False, subscriptionAutoUpdateMode="none",
                        dnsListenAddr="127.0.0.1:52354", logLevel="error")
        api("PUT", "setting", json=settings)
        touch = api("POST", "import", json={"url": "\n".join(lines)})
        (work / "import_result.json").write_text(json.dumps(touch, ensure_ascii=False, indent=2))
        touch = api("GET", "touch")
        print("原生导入返回字段：", list(touch), flush=True)
        # 控制器的touch数据在不同版本中可能位于顶层或touch字段内。
        view = touch.get("touch", touch)
        nodes = view["servers"]
        assert len(nodes) == 54
        print("V2rayA已原生识别54条节点，开始独立连通测试。", flush=True)
        whiches = [{"_type": "server", "id": node["id"], "sub": 0, "outbound": "proxy"} for node in nodes]
        latency = api("GET", "httpLatency", params={"whiches": json.dumps(whiches), "testUrl": "https://huggingface.co/robots.txt"})
        (work / "latency_result.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2))
        touch = api("GET", "touch")
        view = touch.get("touch", touch)
        (work / "tested_nodes.json").write_text(json.dumps(view["servers"], ensure_ascii=False, indent=2))
        for node in view["servers"]:
            print(node["id"], node["name"], node.get("pingLatency"), flush=True)
        ready = [node for node in view["servers"] if re.fullmatch(r"\d+(?:\.\d+)?ms", node.get("pingLatency", ""))]
        ready.sort(key=lambda node: float(re.search(r"[\d.]+", node["pingLatency"]).group()))
        selected = None
        for node in ready[:5]:
            if selected:
                break
            try:
                api("POST", "connection", json={"_type": "server", "id": node["id"], "sub": 0, "outbound": "proxy"})
                api("POST", "v2ray")
                probe = requests.Session()
                probe.trust_env = False
                probe.proxies = {"http": "http://127.0.0.1:21171", "https": "http://127.0.0.1:21171"}
                response = probe.get("https://huggingface.co/robots.txt", timeout=(8,15))
                print("独立代理实际请求：", node["name"], response.status_code, flush=True)
                if response.status_code == 200:
                    selected = node
                    break
            except Exception as exc:
                print("候选节点未通过：", node["name"], type(exc).__name__, flush=True)
            api("DELETE", "v2ray")
            api("DELETE", "connection", json={"_type": "server", "id": node["id"], "sub": 0, "outbound": "proxy"})
        summary = {"work_dir": str(work), "node_count": 54, "passed_latency": len(ready),
                   "selected": selected, "modified_main_v2raya": False, "modified_mihomo": False}
        (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if selected:
            con = sqlite3.connect((config / "v2raya.db").as_uri() + "?mode=ro", uri=True)
            values = dict(con.execute("SELECT key,value FROM system_config WHERE key IN ('system:ports','system:setting','outbound.proxy:connectedServers','system:running')"))
            con.close()
            (work / "selected_settings.json").write_text(json.dumps(values, ensure_ascii=False, indent=2))
        print("暂存验证完成：", json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log.close()


if __name__ == "__main__":
    main()
