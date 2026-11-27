#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка ИСХОДНЫХ WAV корпуса NOIZEUS, без решения лабораторной.

Python 3.11+, стандартная библиотека (ничего устанавливать не нужно).
Источник: https://ecs.utdallas.edu/loizou/speech/noizeus/
Публичное зеркало: https://github.com/chmodsss/noizeus_corpora

Запуск: python download_dataset.py
Для загрузки ВCЕГО NOIZEUS (все восемь типов шума):
    python download_dataset.py --all-noises
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
import time
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "chmodsss/noizeus_corpora"
REF = "master"
ROOT = "NOIZEUS"
NOISE_TYPE = "airport"
VARIANT_TITLE = "Вариант 06: Аэропорт"
SNR_LEVELS = (0, 5, 10, 15)
ALL_NOISE_TYPES = (
    "train", "babble", "car", "restaurant", "street", "airport", "exhibition", "station"
)
HOME = Path(__file__).resolve().parent
USER_AGENT = "University-Lab3-NOIZEUS-Downloader/2.0"


def get_url(url: str, *, max_attempts: int = 3) -> bytes:
    """Загрузить URL с повторными попытками при временных сбоях сети."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json" if "api.github.com" in url else "*/*",
            })
            with urlopen(request, timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in (401, 403, 404):
                break
            if attempt != max_attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Не удалось получить {url}: {last_error}") from last_error


def github_entries(folder: str) -> list[dict]:
    url = f"https://api.github.com/repos/{REPO}/contents/{quote(folder, safe='/')}?ref={REF}"
    response = json.loads(get_url(url))
    if not isinstance(response, list):
        raise ValueError(f"Неверный ответ GitHub при перечислении {folder}: {response}")
    files = [item for item in response if item.get("type") == "file" and item.get("name", "").lower().endswith(".wav")]
    if len(files) != 30:
        raise ValueError(f"Ожидалось 30 WAV в {folder}, но обнаружено {len(files)}; проверьте источник")
    return sorted(files, key=lambda item: item["name"])


def verify_content(content: bytes, meta: dict) -> None:
    if len(content) != meta["size"]:
        raise ValueError(f"Неполная загрузка {meta['name']}: {len(content)} / {meta['size']} байт")
    git_sha = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
    if git_sha != meta["sha"]:
        raise ValueError(f"Несовпадение SHA-1 для {meta['name']}")
    if not (content.startswith(b"RIFF") and content[8:12] == b"WAVE"):
        raise ValueError(f"Полученный файл не является RIFF/WAV: {meta['name']}")


def existing_ok(target: Path, meta: dict) -> bool:
    if not target.is_file() or target.stat().st_size != meta["size"]:
        return False
    # Проверяем хеш содержимого, а не просто расширение WAV.
    with target.open("rb") as f:
        data = f.read()
    try:
        verify_content(data, meta)
    except ValueError:
        return False
    return True


def download_wav(meta: dict, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / meta["name"]
    if existing_ok(target, meta):
        return "уже был"

    # Основной способ — оригинальный двоичный WAV; резерв — base64 из GitHub API.
    direct_url = meta["download_url"]
    try:
        content = get_url(direct_url)
        verify_content(content, meta)
    except Exception as primary_error:
        api_url = meta["url"]
        try:
            info = json.loads(get_url(api_url))
            content = base64.b64decode(info["content"])
            verify_content(content, meta)
        except Exception as secondary_error:
            raise RuntimeError(
                f"Не удалось скачать WAV {meta['name']} напрямую или через GitHub API.\n"
                f"Прямая загрузка: {primary_error}\nGitHub API: {secondary_error}"
            ) from secondary_error

    # Частично загруженный файл никогда не подменит полноценный WAV.
    temporary = target.with_name(target.name + ".part")
    try:
        temporary.write_bytes(content)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "загружен"


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать ВСЕ исходные WAV NOIZEUS без обработки")
    parser.add_argument("--all-noises", action="store_true", help="Весь корпус: 30 чистых + 960 шумных WAV")
    parser.add_argument("--out", type=Path, default=HOME / "dataset", help="Папка выгрузки (по умолчанию ./dataset)")
    args = parser.parse_args()
    destination = args.out.resolve()
    noises = ALL_NOISE_TYPES if args.all_noises else (NOISE_TYPE,)
    expected = 30 * (1 + len(noises) * len(SNR_LEVELS))
    print(f"ЛР №3 — {VARIANT_TITLE}")
    print(f"Папка: {destination}")
    print(f"Будет загружено {expected} исходных WAV; никакой обработки аудио этот скрипт не делает.")

    directories: list[tuple[str, Path]] = [
        (f"{ROOT}/clean_noizeus/wav", destination / "clean")
    ]
    for noise in noises:
        for snr in SNR_LEVELS:
            directories.append((
                f"{ROOT}/{noise}_{snr}dB/wav",
                destination / "noisy" / noise / f"{snr}dB"
            ))

    manifest_rows: list[dict] = []
    errors: list[str] = []
    completed = 0
    for source_dir, local_dir in directories:
        try:
            entries = github_entries(source_dir)
        except Exception as exc:
            print(f"[ОШИБКА] Не удалось получить список файлов: {source_dir}: {exc}", file=sys.stderr)
            errors.append(f"{source_dir}: {exc}")
            continue
        for meta in entries:
            path = local_dir / meta["name"]
            rel = path.relative_to(destination).as_posix()
            try:
                status = download_wav(meta, local_dir)
                completed += 1
                print(f"[{completed:3d}/{expected}] {status}: {rel}")
                manifest_rows.append({
                    "local_path": rel,
                    "source_path": meta["path"],
                    "source_size_bytes": meta["size"],
                    "source_sha1_git": meta["sha"],
                })
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
                print(f"[ОШИБКА] {rel}: {exc}", file=sys.stderr)

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "download_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "local_path", "source_path", "source_size_bytes", "source_sha1_git"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    if errors:
        (destination / "download_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"\nПолучено {completed}/{expected} WAV. Ошибок: {len(errors)}. Перезапустите скрипт для докачки.", file=sys.stderr)
        return 1
    (destination / "download_errors.txt").unlink(missing_ok=True)
    print(f"\nГОТОВО: {completed} WAV. Датасет: {destination}")
    print("Исходные записи не изменялись. Весь код лабораторной работы нужно написать самостоятельно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
