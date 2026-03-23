from __future__ import annotations

import argparse
import importlib.util
import json
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if importlib.util.find_spec("yaml") is not None:
    import yaml
else:
    yaml = None

if importlib.util.find_spec("fitz") is not None:
    import fitz
else:
    fitz = None

if importlib.util.find_spec("pypdf") is not None:
    from pypdf import PdfReader
else:
    PdfReader = None

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "path.yaml"
DEFAULT_TEXT_PREVIEW = 2000
MAX_PDF_SIZE_MB = 256
MAX_STREAM_DECOMPRESS_SIZE = 8 * 1024 * 1024
OBJECT_PATTERN = re.compile(rb"(\d+)\s+(\d+)\s+obj(.*?)endobj", re.DOTALL)
STREAM_PATTERN = re.compile(rb"<<(.*?)>>\s*stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
DICT_PATTERN = re.compile(rb"<<(.*?)>>", re.DOTALL)
REF_PATTERN = re.compile(rb"(\d+)\s+(\d+)\s+R")
FONT_REF_PATTERN = re.compile(rb"/([^\s<>{}\[\]/%]+)\s+(\d+)\s+(\d+)\s+R")
BEGIN_BFCHAR_PATTERN = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
BEGIN_BFRANGE_PATTERN = re.compile(rb"beginbfrange(.*?)endbfrange", re.DOTALL)
BFCHAR_LINE_PATTERN = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
BFRANGE_SIMPLE_PATTERN = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
BFRANGE_ARRAY_PATTERN = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]", re.DOTALL)
HEX_TOKEN_PATTERN = re.compile(rb"<([0-9A-Fa-f\s]+)>")


class PdfProcessError(Exception):
    """PDF 处理失败时抛出的异常。"""


@dataclass
class PathConfig:
    dataset_root: Path


@dataclass
class PdfObject:
    object_id: int
    generation: int
    body: bytes
    dictionary: bytes
    stream: bytes | None


@dataclass
class TextOperand:
    raw_bytes: bytes
    fallback_text: str


@dataclass
class ToUnicodeMap:
    mapping: dict[bytes, str]

    def __post_init__(self) -> None:
        self.code_lengths = sorted({len(key) for key in self.mapping}, reverse=True)

    def decode(self, raw_bytes: bytes) -> str:
        if not raw_bytes:
            return ""

        output: list[str] = []
        index = 0
        while index < len(raw_bytes):
            matched = False
            for code_length in self.code_lengths:
                chunk = raw_bytes[index:index + code_length]
                if chunk in self.mapping:
                    output.append(self.mapping[chunk])
                    index += code_length
                    matched = True
                    break
            if matched:
                continue

            output.append(decode_pdf_bytes(raw_bytes[index:index + 1]))
            index += 1
        return "".join(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查项目目录下的 PDF，并仅在终端输出结果")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="路径配置文件，默认使用 configs/path.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="可选：覆盖配置中的 dataset_root，支持传入任意 PDF 根目录",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_TEXT_PREVIEW,
        help="全文文字在终端中的预览字符数，默认 2000",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="可选：仅处理前 N 个 PDF，便于快速抽查",
    )
    return parser.parse_args()


def clean_inline(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "").strip()
    if text == "None":
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_multiline(text: str) -> str:
    text = str(text or "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned = []
    last_blank = False
    for line in lines:
        if line:
            cleaned.append(line)
            last_blank = False
        else:
            if not last_blank:
                cleaned.append("")
            last_blank = True
    return "\n".join(cleaned).strip()


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"路径配置文件不存在：{config_path}")

    if yaml is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"路径配置文件格式错误：{config_path}")
        return payload

    lines = config_path.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, _, value = line.partition(":")
        if not _:
            raise ValueError(f"无法解析路径配置行：{raw_line}")
        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f"发现未归属分组的缩进行：{raw_line}")
        payload[current_section][key.strip()] = cleaned_value
    return payload


def build_path_config(config_path: Path, input_dir: Path | None) -> PathConfig:
    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        raise ValueError("path.yaml 必须包含 paths 分组")

    config_dir = config_path.expanduser().resolve().parent

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return (config_dir.parent / path).resolve()

    dataset_root = input_dir.expanduser().resolve() if input_dir is not None else resolve_path(str(paths_payload["dataset_root"]))
    return PathConfig(dataset_root=dataset_root)


def iter_pdf_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def options_to_dict(opt: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not opt:
        return mapping
    for item in opt:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            code = clean_inline(item[0])
            text = clean_inline(item[1])
            mapping[code] = text
        else:
            key = clean_inline(item)
            mapping[key] = key
    return mapping


def extract_with_pypdf(pdf_path: Path) -> tuple[list[tuple[str, str]], str, int]:
    if PdfReader is None:
        raise PdfProcessError("当前环境未安装 pypdf，无法使用 pypdf 路径提取")

    reader = PdfReader(str(pdf_path))
    fields = reader.get_fields() or {}
    ordered_fields: list[tuple[str, str]] = []
    for field_name, field in fields.items():
        value = clean_inline(field.get("/V"))
        opt_map = options_to_dict(field.get("/Opt"))
        display_value = opt_map.get(value, value) if opt_map else value
        ordered_fields.append((field_name, display_value))
    return ordered_fields, "", len(reader.pages)


def extract_with_fitz(pdf_path: Path) -> str:
    if fitz is None:
        raise PdfProcessError("当前环境未安装 fitz，无法使用 PyMuPDF 路径提取")

    doc = fitz.open(str(pdf_path))
    page_texts = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = clean_multiline(page.get_text("text", sort=True))
            if text:
                page_texts.append(f"[第{page_number}页]\n{text}")
    finally:
        doc.close()
    return "\n\n".join(page_texts).strip()


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

    if data.startswith((b"\xfe\xff", b"\xff\xfe")):
        for encoding in ("utf-16", "utf-16-be", "utf-16-le"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue

    cleaned = data.replace(b"\x00", b"")
    for encoding in ("utf-8", "utf-16-be", "gb18030", "gbk", "big5", "latin-1"):
        try:
            return cleaned.decode(encoding)
        except UnicodeDecodeError:
            continue
    return cleaned.decode("latin-1", errors="ignore")


def decode_pdf_literal_bytes(data: bytes) -> bytes:
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

    return bytes(result)


def decode_pdf_literal_string(data: bytes) -> str:
    return decode_pdf_bytes(decode_pdf_literal_bytes(data)).strip()


def decode_pdf_hex_bytes(data: bytes) -> bytes:
    cleaned = re.sub(rb"\s+", b"", data)
    if len(cleaned) % 2 == 1:
        cleaned += b"0"
    try:
        return bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return cleaned


def decode_pdf_hex_string(data: bytes) -> str:
    return decode_pdf_bytes(decode_pdf_hex_bytes(data)).strip()


def parse_pdf_string(data: bytes, start: int) -> tuple[str, int]:
    operand, next_index = parse_pdf_text_operand(data, start)
    return operand.fallback_text, next_index


def parse_pdf_text_operand(data: bytes, start: int) -> tuple[TextOperand, int]:
    if start >= len(data):
        return TextOperand(raw_bytes=b"", fallback_text=""), start

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
        raw_bytes = decode_pdf_literal_bytes(bytes(chunk))
        return TextOperand(raw_bytes=raw_bytes, fallback_text=decode_pdf_bytes(raw_bytes).strip()), index

    if marker == b"<" and data[start:start + 2] != b"<<":
        end = data.find(b">", start + 1)
        if end == -1:
            return TextOperand(raw_bytes=b"", fallback_text=""), len(data)
        raw_bytes = decode_pdf_hex_bytes(data[start + 1:end])
        return TextOperand(raw_bytes=raw_bytes, fallback_text=decode_pdf_bytes(raw_bytes).strip()), end + 1

    return TextOperand(raw_bytes=b"", fallback_text=""), start


def extract_dict_value(dictionary: bytes, key: bytes) -> str:
    match = re.search(rb"/" + re.escape(key) + rb"\b", dictionary)
    if not match:
        return ""

    index = match.end()
    while index < len(dictionary) and dictionary[index:index + 1] in b"\x00 \t\r\n":
        index += 1

    marker = dictionary[index:index + 1]
    if marker in {b"(", b"<"}:
        value, _ = parse_pdf_string(dictionary, index)
        return value.strip()
    if marker == b"/":
        end = index + 1
        while end < len(dictionary) and dictionary[end:end + 1] not in b"/[]()<>\r\n\t ":
            end += 1
        return decode_pdf_name(dictionary[index + 1:end]).strip()
    return ""


def extract_single_ref(dictionary: bytes, key: bytes) -> int | None:
    match = re.search(rb"/" + re.escape(key) + rb"\s+(\d+)\s+(\d+)\s+R", dictionary)
    if not match:
        return None
    return int(match.group(1))


def extract_all_refs(dictionary: bytes, key: bytes) -> list[int]:
    direct_match = re.search(rb"/" + re.escape(key) + rb"\s+(\d+)\s+(\d+)\s+R", dictionary)
    if direct_match:
        return [int(direct_match.group(1))]

    array_match = re.search(rb"/" + re.escape(key) + rb"\s*\[(.*?)\]", dictionary, re.DOTALL)
    if not array_match:
        return []
    return [int(match.group(1)) for match in REF_PATTERN.finditer(array_match.group(1))]


def safe_decompress_stream(stream_data: bytes, max_output_size: int) -> bytes:
    decompressor = zlib.decompressobj()
    output_chunks: list[bytes] = []
    total_size = 0
    remaining = stream_data

    while remaining:
        chunk = decompressor.decompress(remaining, max_output_size - total_size)
        if chunk:
            output_chunks.append(chunk)
            total_size += len(chunk)
        remaining = decompressor.unconsumed_tail

        if total_size >= max_output_size:
            raise PdfProcessError(
                f"解压后的流超过 {max_output_size // (1024 * 1024)} MB，已停止处理以避免内存占用过高"
            )

        if not remaining:
            break

    tail = decompressor.flush(max_output_size - total_size)
    if tail:
        output_chunks.append(tail)
        total_size += len(tail)

    if total_size >= max_output_size:
        raise PdfProcessError(
            f"解压后的流超过 {max_output_size // (1024 * 1024)} MB，已停止处理以避免内存占用过高"
        )

    return b"".join(output_chunks)


def maybe_decompress_stream(stream_dict: bytes, stream_data: bytes) -> bytes:
    filters = stream_dict.decode("latin-1", errors="ignore")
    if "/FlateDecode" in filters:
        try:
            return safe_decompress_stream(stream_data, MAX_STREAM_DECOMPRESS_SIZE)
        except (zlib.error, PdfProcessError):
            return b""
    return stream_data


def parse_pdf_objects(pdf_bytes: bytes) -> dict[int, PdfObject]:
    objects: dict[int, PdfObject] = {}
    for match in OBJECT_PATTERN.finditer(pdf_bytes):
        object_id = int(match.group(1))
        generation = int(match.group(2))
        body = match.group(3).strip()

        dictionary = b""
        stream: bytes | None = None
        stream_match = STREAM_PATTERN.search(body)
        if stream_match:
            dictionary = stream_match.group(1)
            stream = maybe_decompress_stream(stream_match.group(1), stream_match.group(2))
        else:
            dict_match = DICT_PATTERN.search(body)
            if dict_match:
                dictionary = dict_match.group(1)

        objects[object_id] = PdfObject(
            object_id=object_id,
            generation=generation,
            body=body,
            dictionary=dictionary,
            stream=stream,
        )
    return objects


def extract_form_fields(objects: dict[int, PdfObject]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    seen_keys: set[str] = set()

    for object_id in sorted(objects):
        dictionary = objects[object_id].dictionary
        if not dictionary or b"/T" not in dictionary:
            continue

        key = extract_dict_value(dictionary, b"T") or extract_dict_value(dictionary, b"TU")
        if not key or key in seen_keys:
            continue

        value = (
            extract_dict_value(dictionary, b"V")
            or extract_dict_value(dictionary, b"DV")
            or extract_dict_value(dictionary, b"RV")
            or extract_dict_value(dictionary, b"AS")
        )
        fields.append((key, normalize_value(value)))
        seen_keys.add(key)

    return fields


def parse_tounicode_cmap(stream_data: bytes) -> ToUnicodeMap:
    mapping: dict[bytes, str] = {}

    for block in BEGIN_BFCHAR_PATTERN.finditer(stream_data):
        for item in BFCHAR_LINE_PATTERN.finditer(block.group(1)):
            src = decode_pdf_hex_bytes(item.group(1))
            dst = decode_pdf_hex_string(item.group(2))
            if src and dst:
                mapping[src] = dst

    for block in BEGIN_BFRANGE_PATTERN.finditer(stream_data):
        content = block.group(1)
        for item in BFRANGE_SIMPLE_PATTERN.finditer(content):
            start = int(item.group(1), 16)
            end = int(item.group(2), 16)
            dst_start = int(item.group(3), 16)
            width = max(1, len(item.group(1)) // 2)
            dst_width = max(1, len(item.group(3)) // 2)
            for offset, code_point in enumerate(range(start, end + 1)):
                src = code_point.to_bytes(width, byteorder="big")
                dst = (dst_start + offset).to_bytes(dst_width, byteorder="big")
                mapping[src] = decode_pdf_bytes(dst)

        for item in BFRANGE_ARRAY_PATTERN.finditer(content):
            start = int(item.group(1), 16)
            end = int(item.group(2), 16)
            width = max(1, len(item.group(1)) // 2)
            destinations = [decode_pdf_hex_string(match.group(1)) for match in HEX_TOKEN_PATTERN.finditer(item.group(3))]
            for offset, code_point in enumerate(range(start, end + 1)):
                if offset >= len(destinations):
                    break
                src = code_point.to_bytes(width, byteorder="big")
                mapping[src] = destinations[offset]

    return ToUnicodeMap(mapping=mapping)


def build_font_cmaps(objects: dict[int, PdfObject]) -> dict[int, ToUnicodeMap]:
    font_cmaps: dict[int, ToUnicodeMap] = {}
    for object_id, pdf_object in objects.items():
        if not pdf_object.dictionary or b"/ToUnicode" not in pdf_object.dictionary:
            continue
        to_unicode_ref = extract_single_ref(pdf_object.dictionary, b"ToUnicode")
        if to_unicode_ref is None or to_unicode_ref not in objects:
            continue
        cmap_stream = objects[to_unicode_ref].stream
        if not cmap_stream:
            continue
        cmap = parse_tounicode_cmap(cmap_stream)
        if cmap.mapping:
            font_cmaps[object_id] = cmap
    return font_cmaps


def resolve_font_resources(resource_dict: bytes, objects: dict[int, PdfObject]) -> dict[str, int]:
    font_map: dict[str, int] = {}
    if not resource_dict:
        return font_map

    font_block_match = re.search(rb"/Font\s*<<(.*?)>>", resource_dict, re.DOTALL)
    if not font_block_match:
        font_ref = extract_single_ref(resource_dict, b"Font")
        if font_ref is not None and font_ref in objects:
            return resolve_font_resources(objects[font_ref].dictionary, objects)
        return font_map

    font_block = font_block_match.group(1)
    for match in FONT_REF_PATTERN.finditer(font_block):
        alias = decode_pdf_name(match.group(1))
        font_map[alias] = int(match.group(2))
    return font_map


def decode_text_operand(operand: TextOperand, cmap: ToUnicodeMap | None) -> str:
    if cmap is not None:
        decoded = normalize_value(cmap.decode(operand.raw_bytes))
        if decoded:
            return decoded
    return normalize_value(operand.fallback_text)


def skip_whitespace(data: bytes, index: int) -> int:
    while index < len(data):
        current = data[index:index + 1]
        if current in b" \t\r\n\x0c\x00":
            index += 1
            continue
        if current == b"%":
            while index < len(data) and data[index:index + 1] not in {b"\r", b"\n"}:
                index += 1
            continue
        break
    return index


def parse_name_token(data: bytes, start: int) -> tuple[str, int]:
    end = start + 1
    while end < len(data) and data[end:end + 1] not in b" \t\r\n[]()<>{}/%":
        end += 1
    return decode_pdf_name(data[start + 1:end]), end


def parse_array_token(data: bytes, start: int) -> tuple[list[object], int]:
    items: list[object] = []
    index = start + 1
    while index < len(data):
        index = skip_whitespace(data, index)
        if index >= len(data):
            break
        if data[index:index + 1] == b"]":
            return items, index + 1
        token, index = parse_operand_token(data, index)
        if token is None:
            index += 1
            continue
        items.append(token)
    return items, index


def parse_operand_token(data: bytes, start: int) -> tuple[object | None, int]:
    index = skip_whitespace(data, start)
    if index >= len(data):
        return None, index

    current = data[index:index + 1]
    if current == b"/":
        return parse_name_token(data, index)
    if current in {b"(", b"<"} and data[index:index + 2] != b"<<":
        return parse_pdf_text_operand(data, index)
    if current == b"[":
        return parse_array_token(data, index)

    end = index
    while end < len(data) and data[end:end + 1] not in b" \t\r\n[]()<>{}/%":
        end += 1
    token = data[index:end].decode("latin-1", errors="ignore")
    return token, end


def extract_text_from_stream(stream_data: bytes, font_alias_to_obj: dict[str, int], font_cmaps: dict[int, ToUnicodeMap]) -> str:
    snippets: list[str] = []
    operands: list[object] = []
    current_font: str | None = None
    index = 0

    while index < len(stream_data):
        index = skip_whitespace(stream_data, index)
        if index >= len(stream_data):
            break

        token, next_index = parse_operand_token(stream_data, index)
        if token is None:
            index += 1
            continue

        if isinstance(token, str) and token:
            operator = token
            cmap = font_cmaps.get(font_alias_to_obj.get(current_font or "", -1))
            if operator == "Tf" and len(operands) >= 2 and isinstance(operands[-2], str):
                current_font = operands[-2]
            elif operator in {"Tj", "'", '"'} and operands and isinstance(operands[-1], TextOperand):
                text = decode_text_operand(operands[-1], cmap)
                if text:
                    snippets.append(text)
            elif operator == "TJ" and operands and isinstance(operands[-1], list):
                parts: list[str] = []
                for item in operands[-1]:
                    if isinstance(item, TextOperand):
                        decoded = decode_text_operand(item, cmap)
                        if decoded:
                            parts.append(decoded)
                text = normalize_value("".join(parts))
                if text:
                    snippets.append(text)
            operands = []
        else:
            operands.append(token)
        index = next_index

    compact_lines = [normalize_value(item) for item in snippets if normalize_value(item)]
    return "\n".join(compact_lines).strip()


def extract_text(objects: dict[int, PdfObject], preview_chars: int | None = None) -> str:
    font_cmaps = build_font_cmaps(objects)
    page_texts: list[str] = []
    seen_stream_ids: set[int] = set()

    for object_id in sorted(objects):
        pdf_object = objects[object_id]
        dictionary = pdf_object.dictionary
        if not dictionary or b"/Type /Page" not in dictionary:
            continue

        resource_ref = extract_single_ref(dictionary, b"Resources")
        resource_dict = objects[resource_ref].dictionary if resource_ref in objects else dictionary
        if resource_ref is None:
            inline_match = re.search(rb"/Resources\s*<<(.*?)>>", dictionary, re.DOTALL)
            if inline_match:
                resource_dict = inline_match.group(1)

        font_alias_to_obj = resolve_font_resources(resource_dict, objects)
        content_refs = extract_all_refs(dictionary, b"Contents")
        if not content_refs and pdf_object.stream:
            content_refs = [object_id]

        page_snippets: list[str] = []
        for content_ref in content_refs:
            content_object = objects.get(content_ref)
            if content_object is None or not content_object.stream:
                continue
            extracted = extract_text_from_stream(content_object.stream, font_alias_to_obj, font_cmaps)
            if extracted:
                page_snippets.append(extracted)
                seen_stream_ids.add(content_ref)

        if page_snippets:
            page_texts.append("\n".join(page_snippets))

    if not page_texts:
        fallback_texts: list[str] = []
        for object_id in sorted(objects):
            pdf_object = objects[object_id]
            if object_id in seen_stream_ids or not pdf_object.stream:
                continue
            extracted = extract_text_from_stream(pdf_object.stream, {}, font_cmaps)
            if extracted:
                fallback_texts.append(extracted)
        page_texts = fallback_texts

    full_text = "\n".join(page_texts).strip()
    if preview_chars is not None and preview_chars > 0 and len(full_text) > preview_chars:
        return full_text[:preview_chars].rstrip() + "\n……（全文过长，已截断显示）"
    return full_text


def fallback_extract(pdf_path: Path, preview_chars: int) -> tuple[list[tuple[str, str]], str, int | str]:
    pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if pdf_size_mb > MAX_PDF_SIZE_MB:
        raise PdfProcessError(
            f"文件大小为 {pdf_size_mb:.1f} MB，超过限制 {MAX_PDF_SIZE_MB} MB，已跳过以避免内存占用过高"
        )

    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        raise PdfProcessError(f"读取文件失败：{exc}") from exc

    if not pdf_bytes.startswith(b"%PDF"):
        raise PdfProcessError("文件头不是有效的 PDF 标识")

    objects = parse_pdf_objects(pdf_bytes)
    if not objects:
        raise PdfProcessError("未解析到 PDF 对象，可能是加密或结构异常文件")

    fields = extract_form_fields(objects)
    full_text = extract_text(objects, preview_chars=preview_chars)
    page_count = sum(1 for obj in objects.values() if obj.dictionary and b"/Type /Page" in obj.dictionary)
    return fields, full_text, page_count or ""


def process_single_pdf(pdf_path: Path, preview_chars: int) -> tuple[list[tuple[str, str]], str, int | str, str]:
    strategy_parts: list[str] = []
    fields: list[tuple[str, str]] = []
    full_text = ""
    page_count: int | str = ""

    if PdfReader is not None:
        fields, _, page_count = extract_with_pypdf(pdf_path)
        strategy_parts.append("pypdf表单")
    else:
        strategy_parts.append("内置表单回退")

    if fitz is not None:
        full_text = extract_with_fitz(pdf_path)
        if preview_chars > 0 and len(full_text) > preview_chars:
            full_text = full_text[:preview_chars].rstrip() + "\n……（全文过长，已截断显示）"
        strategy_parts.append("fitz正文")
    else:
        strategy_parts.append("内置正文回退")

    if PdfReader is None or fitz is None:
        fallback_fields, fallback_text, fallback_pages = fallback_extract(pdf_path, preview_chars)
        if not fields:
            fields = fallback_fields
        if not full_text:
            full_text = fallback_text
        if page_count == "":
            page_count = fallback_pages

    return fields, full_text, page_count, " + ".join(strategy_parts)


def print_pdf_result(
    pdf_path: Path,
    fields: list[tuple[str, str]],
    full_text: str,
    page_count: int | str,
    strategy: str,
) -> None:
    print(f"\n{'=' * 120}")
    print(f"文件：{pdf_path.name}")
    print(f"路径：{pdf_path}")
    print(f"页数：{page_count if page_count != '' else '未知'}")
    print(f"提取方式：{strategy}")

    print("\n【表单字段】")
    if fields:
        ordered_fields = OrderedDict(fields)
        for key, value in ordered_fields.items():
            print(f"{key}: {value if value else '（空值）'}")
    else:
        print("未提取到表单字段")

    print("\n【全文文字预览】")
    if full_text:
        print(full_text)
    else:
        print("未提取到可读文字")


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir)
    input_dir = path_config.dataset_root

    if not input_dir.exists():
        print(f"输入目录不存在：{input_dir}")
        return
    if not input_dir.is_dir():
        print(f"输入路径不是目录：{input_dir}")
        return

    pdf_files = list(iter_pdf_files(input_dir))
    if args.max_files is not None and args.max_files > 0:
        pdf_files = pdf_files[:args.max_files]

    if not pdf_files:
        print(f"目录下未找到 PDF 文件：{input_dir}")
        return

    success_count = 0
    failed_count = 0

    runtime_info = {
        "输入目录": str(input_dir),
        "PDF数量": len(pdf_files),
        "全文预览字符数": args.preview_chars,
        "已安装fitz": fitz is not None,
        "已安装pypdf": PdfReader is not None,
        "输出文件": "不生成，仅终端输出",
    }
    print("开始检查 PDF：")
    print(json.dumps(runtime_info, ensure_ascii=False, indent=2))

    for pdf_path in pdf_files:
        try:
            fields, full_text, page_count, strategy = process_single_pdf(pdf_path, args.preview_chars)
            print_pdf_result(pdf_path, fields, full_text, page_count, strategy)
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"\n{'=' * 120}")
            print(f"文件：{pdf_path.name}")
            print(f"路径：{pdf_path}")
            print(f"状态：处理失败")
            print(f"原因：{exc}")

    print(f"\n{'=' * 120}")
    print("处理完成：")
    print(
        json.dumps(
            {
                "输入目录": str(input_dir),
                "总 PDF 数量": len(pdf_files),
                "成功数量": success_count,
                "失败数量": failed_count,
                "输出文件": "无，仅终端输出",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
