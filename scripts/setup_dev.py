"""
Готовит воспроизводимое окружение стенда на Windows и Linux из lock-файлов.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(environment_file: Path) -> None:
    """
    Устанавливает зависимости и создаёт настройки, не перезаписывая существующие.
    """
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Запустите setup_dev.py через Python 3.12")
    pnpm = shutil.which("pnpm")
    node = shutil.which("node")
    if pnpm is None or node is None:
        raise RuntimeError("Установите Node.js >=22.12 и pnpm 11.19.0 в PATH")
    version = subprocess.check_output([node, "--version"], text=True).strip().lstrip("v")
    if tuple(map(int, version.split("."))) < (22, 12, 0):
        raise RuntimeError("Требуется Node.js >=22.12")
    expected = json.loads((ROOT / "Frontend/package.json").read_text())["packageManager"]
    actual = subprocess.check_output([pnpm, "--version"], text=True).strip()
    if expected != f"pnpm@{actual}":
        raise RuntimeError(f"Требуется {expected}")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    venv_version = subprocess.check_output(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
        text=True,
    ).strip()
    if venv_version != "3.12":
        raise RuntimeError("Существующая .venv использует другой Python; создайте новую вручную")
    subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements.lock.txt"],
                   cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], cwd=ROOT, check=True)
    subprocess.run([pnpm, "install", "--frozen-lockfile"], cwd=ROOT / "Frontend", check=True)
    subprocess.run([pnpm, "build"], cwd=ROOT / "Frontend", check=True)
    if environment_file.exists():
        print("Файл окружения уже существует; его содержимое сохранено.")
    else:
        template = (ROOT / ".env.example").read_text(encoding="utf-8")
        template = template.replace("SERVICE_AUTH_TOKEN=\n",
                                    f"SERVICE_AUTH_TOKEN={secrets.token_urlsafe(32)}\n")
        # O_EXCL защищает от перезаписи, режим 0600 ограничивает доступ на POSIX.
        descriptor = os.open(environment_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(template)
        print("Создан файл окружения с новым сервисным токеном; заполните настройки LLM.")


def main() -> int:
    """
    Принимает путь настроек и возвращает ненулевой код при ошибке подготовки.
    """
    parser = argparse.ArgumentParser(description="Подготовка локального дев-стенда")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    arguments = parser.parse_args()
    try:
        prepare(arguments.env_file.resolve())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Подготовка не завершена: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
