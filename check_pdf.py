from __future__ import annotations

import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INPUT_DIR = Path("/home/Lim/datasets/project4/ZS08004085/ZS0048989881/pdf")
MAX_TEXT_PREVIEW = 4000
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


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def iter_pdf_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


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


def extract_text(objects: dict[int, PdfObject]) -> str:
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
            display_value = value if value else "（空值）"
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
    full_text = extract_text(objects)
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
