from __future__ import annotations

import re
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

INPUT_DIR = Path("/home/Lim/datasets/project4/ZS08004085/ZS0048989881/pdf")
MAX_TEXT_PREVIEW = 2000

STREAM_PATTERN = re.compile(
    rb"<<(.*?)>>\s*stream\r?\n(.*?)\r?\nendstream", re.DOTALL
)
DICT_PATTERN = re.compile(rb"<<(.*?)>>", re.DOTALL)
BT_ET_PATTERN = re.compile(rb"BT(.*?)ET", re.DOTALL)
HEX_TEXT_PATTERN = re.compile(rb"<([0-9A-Fa-f\s]+)>")


class PdfProcessError(Exception):
    """PDF 处理失败时抛出的异常。"""


def decode_pdf_name(name: bytes) -> str:
    parts: list[str] = []
    index = 0
    while index < len(name):
        if name[index:index + 1] == b"#" and index + 2 < len(name):
            hex_part = name[index + 1:index + 3]
            try:
                parts.append(bytes.fromhex(hex_part.decode("ascii")).decode("latin-1"))
                index += 3
                continue
            except ValueError:
                pass
        parts.append(chr(name[index]))
        index += 1
    return "".join(parts)


def decode_pdf_bytes(data: bytes) -> str:
    if not data:
        return ""

    cleaned = data.replace(b"\x00", b"")
    for encoding in ("utf-8", "utf-16", "utf-16-be", "gb18030", "latin-1"):
        try:
            return cleaned.decode(encoding)
        except UnicodeDecodeError:
            continue
    return cleaned.decode("latin-1", errors="ignore")


def decode_pdf_literal_string(data: bytes) -> str:
    result = bytearray()
    index = 0
    while index < len(data):
        current = data[index]
        if current != 0x5C:
            result.append(current)
            index += 1
            continue

        index += 1
        if index >= len(data):
            break
        escaped = data[index]
        mapping = {
            ord("n"): b"\n",
            ord("r"): b"\r",
            ord("t"): b"\t",
            ord("b"): b"\b",
            ord("f"): b"\f",
            ord("("): b"(",
            ord(")"): b")",
            ord("\\"): b"\\",
        }
        if escaped in mapping:
            result.extend(mapping[escaped])
            index += 1
            continue

        if escaped in (ord("\n"), ord("\r")):
            if escaped == ord("\r") and index + 1 < len(data) and data[index + 1] == ord("\n"):
                index += 2
            else:
                index += 1
            continue

        if 48 <= escaped <= 55:
            octal_digits = bytes([escaped])
            index += 1
            for _ in range(2):
                if index < len(data) and 48 <= data[index] <= 55:
                    octal_digits += bytes([data[index]])
                    index += 1
                else:
                    break
            result.append(int(octal_digits, 8))
            continue

        result.append(escaped)
        index += 1

    return decode_pdf_bytes(bytes(result)).strip()


def decode_pdf_hex_string(data: bytes) -> str:
    cleaned = re.sub(rb"\s+", b"", data)
    if len(cleaned) % 2 == 1:
        cleaned += b"0"
    try:
        raw = bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return decode_pdf_bytes(cleaned)
    return decode_pdf_bytes(raw).strip()


def parse_pdf_string(data: bytes, start: int) -> tuple[str, int]:
    if start >= len(data):
        return "", start

    marker = data[start:start + 1]
    if marker == b"(":
        depth = 1
        index = start + 1
        chunk = bytearray()
        while index < len(data) and depth > 0:
            current = data[index]
            if current == 0x5C:
                if index + 1 < len(data):
                    chunk.extend(data[index:index + 2])
                    index += 2
                else:
                    chunk.append(current)
                    index += 1
                continue
            if current == 0x28:
                depth += 1
                chunk.append(current)
                index += 1
                continue
            if current == 0x29:
                depth -= 1
                if depth == 0:
                    index += 1
                    break
                chunk.append(current)
                index += 1
                continue
            chunk.append(current)
            index += 1
        return decode_pdf_literal_string(bytes(chunk)), index

    if marker == b"<" and data[start:start + 2] != b"<<":
        end = data.find(b">", start + 1)
        if end == -1:
            return "", len(data)
        return decode_pdf_hex_string(data[start + 1:end]), end + 1

    if marker == b"/":
        end = start + 1
        while end < len(data) and data[end:end + 1] not in b"/[]()<>\r\n\t ":
            end += 1
        return decode_pdf_name(data[start + 1:end]), end

    return "", start


def extract_dict_value(dictionary: bytes, key: bytes) -> str:
    match = re.search(rb"/" + re.escape(key) + rb"\b", dictionary)
    if not match:
        return ""

    index = match.end()
    while index < len(dictionary) and dictionary[index:index + 1] in b"\x00 \t\r\n":
        index += 1
    value, _ = parse_pdf_string(dictionary, index)
    return value.strip()


def normalize_value(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if compact in {"", "Off", "Yes"}:
        return compact
    return compact


def iter_pdf_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def extract_form_fields(pdf_bytes: bytes) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for match in DICT_PATTERN.finditer(pdf_bytes):
        dictionary = match.group(1)
        if b"/T" not in dictionary:
            continue

        key = extract_dict_value(dictionary, b"T") or extract_dict_value(dictionary, b"TU")
        if not key:
            continue

        value = (
            extract_dict_value(dictionary, b"V")
            or extract_dict_value(dictionary, b"DV")
            or extract_dict_value(dictionary, b"AS")
        )
        normalized = normalize_value(value)
        pair = (key, normalized)
        if pair in seen:
            continue
        seen.add(pair)
        fields.append(pair)

    return fields


def maybe_decompress_stream(stream_dict: bytes, stream_data: bytes) -> bytes:
    filters = stream_dict.decode("latin-1", errors="ignore")
    if "/FlateDecode" in filters:
        try:
            return zlib.decompress(stream_data)
        except zlib.error:
            return b""
    return stream_data


def extract_strings_from_tj_array(array_content: bytes) -> str:
    parts: list[str] = []
    index = 0
    while index < len(array_content):
        current = array_content[index:index + 1]
        if current in {b"(", b"<"}:
            text, next_index = parse_pdf_string(array_content, index)
            if text:
                parts.append(text)
            index = max(next_index, index + 1)
            continue
        index += 1
    return "".join(parts).strip()


def extract_text_from_stream(stream_data: bytes) -> str:
    snippets: list[str] = []
    for block in BT_ET_PATTERN.finditer(stream_data):
        content = block.group(1)

        index = 0
        while index < len(content):
            current = content[index:index + 1]
            if current in {b"(", b"<"}:
                text, next_index = parse_pdf_string(content, index)
                tail = content[next_index:next_index + 4]
                if b"Tj" in tail or b"'" in tail or b'"' in tail:
                    if text:
                        snippets.append(text)
                index = max(next_index, index + 1)
                continue

            if current == b"[":
                array_end = content.find(b"]", index + 1)
                if array_end == -1:
                    break
                tail = content[array_end + 1:array_end + 6]
                if b"TJ" in tail:
                    text = extract_strings_from_tj_array(content[index + 1:array_end])
                    if text:
                        snippets.append(text)
                index = array_end + 1
                continue
            index += 1

    compact_lines = [re.sub(r"\s+", " ", item).strip() for item in snippets if item.strip()]
    return "\n".join(compact_lines).strip()


def extract_text(pdf_bytes: bytes) -> str:
    texts: list[str] = []
    seen: set[str] = set()

    for stream_match in STREAM_PATTERN.finditer(pdf_bytes):
        stream_dict = stream_match.group(1)
        stream_data = stream_match.group(2)
        decoded_stream = maybe_decompress_stream(stream_dict, stream_data)
        if not decoded_stream:
            continue
        extracted = extract_text_from_stream(decoded_stream)
        if extracted and extracted not in seen:
            texts.append(extracted)
            seen.add(extracted)

    full_text = "\n".join(texts).strip()
    if len(full_text) > MAX_TEXT_PREVIEW:
        return full_text[:MAX_TEXT_PREVIEW].rstrip() + "\n……（全文过长，已截断显示）"
    return full_text


def print_pdf_result(pdf_path: Path, fields: list[tuple[str, str]], full_text: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"文件: {pdf_path.name}")
    print(f"路径: {pdf_path}")
    print(f"{'-' * 80}")

    if fields:
        print("表单字段:")
        ordered_fields = OrderedDict(fields)
        for key, value in ordered_fields.items():
            display_value = value if value != "" else "（空值）"
            print(f"{key}: {display_value}")
    else:
        print("表单字段: 未提取到表单字段")

    print(f"{'-' * 80}")
    if full_text:
        print("全文文字:")
        print(full_text)
    else:
        print("全文文字: 未提取到可读文字")


def process_single_pdf(pdf_path: Path) -> tuple[list[tuple[str, str]], str]:
    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        raise PdfProcessError(f"读取文件失败：{exc}") from exc

    if not pdf_bytes.startswith(b"%PDF"):
        raise PdfProcessError("文件头不是有效的 PDF 标识")

    fields = extract_form_fields(pdf_bytes)
    full_text = extract_text(pdf_bytes)
    return fields, full_text


def main() -> None:
    input_dir = INPUT_DIR.expanduser()
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    if not input_dir.is_dir():
        print(f"输入路径不是目录: {input_dir}")
        return

    pdf_files = list(iter_pdf_files(input_dir))
    if not pdf_files:
        print(f"目录下未找到 PDF 文件: {input_dir}")
        return

    total_count = len(pdf_files)
    success_count = 0
    failed_count = 0

    print(f"开始检查 PDF，目录: {input_dir}")
    print(f"共发现 PDF 文件: {total_count}")

    for pdf_path in pdf_files:
        try:
            fields, full_text = process_single_pdf(pdf_path)
            print_pdf_result(pdf_path, fields, full_text)
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"\n{'=' * 80}")
            print(f"文件: {pdf_path.name}")
            print(f"错误: 处理失败，已跳过。原因: {exc}")

    print(f"\n{'=' * 80}")
    print("处理完成")
    print(f"总 PDF 数量: {total_count}")
    print(f"成功处理数量: {success_count}")
    print(f"失败数量: {failed_count}")


if __name__ == "__main__":
    main()
