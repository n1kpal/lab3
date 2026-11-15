"""Добавляет src/ в sys.path"""
import pathlib
import sys

src = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src))