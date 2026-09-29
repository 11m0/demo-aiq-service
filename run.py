"""
Запускает весь локальный стенд из PyCharm или командной строки.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = PROJECT_ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
VITE_SCRIPT = PROJECT_ROOT / "Frontend" / "node_modules" / "vite" / "bin" / "vite.js"
ENVIRONMENT_FILE = PROJECT_ROOT / ".env"

API_SOURCE_DIRECTORIES = (
    PROJECT_ROOT / "Agent" / "src",
    PROJECT_ROOT / "API" / "src",
    PROJECT_ROOT / "Core" / "src",
    PROJECT_ROOT / "Contracts" / "src",
)
BACKEND_SOURCE_DIRECTORIES = (
    PROJECT_ROOT / "Backend" / "src",
    PROJECT_ROOT / "Core" / "src",
    PROJECT_ROOT / "Contracts" / "src",
)
ALL_SOURCE_DIRECTORIES = tuple(
    dict.fromkeys((*API_SOURCE_DIRECTORIES, *BACKEND_SOURCE_DIRECTORIES))
)

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _IOCounters(ctypes.Structure):
    """
    Повторяет структуру счётчиков ввода-вывода Windows Job Object.
    """

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    """
    Описывает основные ограничения Windows Job Object.
    """

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    """
    Добавляет к Job Object флаг уничтожения процессов при закрытии handle.
    """

    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsProcessJob:
    """
    Гарантирует остановку всего дерева процессов вместе с launcher-процессом.
    """

    def __init__(self) -> None:
        """
        Создаёт Job Object с политикой KILL_ON_JOB_CLOSE.
        """
        if os.name != "nt":
            raise RuntimeError("Единый локальный запуск сейчас поддерживает Windows")

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_function_signatures()
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle: int | None = int(handle)

        information = _JobObjectExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        configured = self._kernel32.SetInformationJobObject(
            wintypes.HANDLE(self._handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )
        if not configured:
            error_code = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error_code)

    def _configure_function_signatures(self) -> None:
        """
        Задаёт ctypes точные типы функций Windows API.
        """
        self._kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
        ]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        """
        Добавляет запущенный процесс и будущих потомков в Job Object.
        """
        if self._handle is None:
            raise RuntimeError("Windows Job Object уже закрыт")
        process_handle = self._kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            process.pid,
        )
        if not process_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            assigned = self._kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(self._handle),
                process_handle,
            )
            if not assigned:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._kernel32.CloseHandle(process_handle)

    def close(self) -> None:
        """
        Закрывает Job Object и тем самым завершает все его процессы.
        """
        if self._handle is None:
            return
        self._kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None


class PosixProcessJob:
    """
    Завершает отдельные группы процессов стенда на Linux и macOS.
    """

    def __init__(self) -> None:
        """
        Создаёт список управляемых групп процессов.
        """
        self._processes: list[subprocess.Popen[bytes]] = []

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        """
        Запоминает процесс, запущенный в отдельной POSIX-сессии.
        """
        self._processes.append(process)

    def close(self) -> None:
        """
        Даёт сервисам время на остановку, затем удаляет оставшиеся процессы группы.
        """
        for process in self._processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for process in self._processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._processes.clear()


@dataclass(frozen=True, slots=True)
class ManagedProcess:
    """
    Связывает понятное имя компонента с его системным процессом.
    """

    name: str
    process: subprocess.Popen[bytes]


def validate_environment(environment_file: Path = ENVIRONMENT_FILE) -> str:
    """
    Проверяет локальные зависимости и возвращает путь к Node.js.
    """
    if not PYTHON_EXECUTABLE.is_file():
        raise RuntimeError(f"Не найден Python проекта: {PYTHON_EXECUTABLE}")
    if not VITE_SCRIPT.is_file():
        raise RuntimeError("Не найден Vite. Выполните 'pnpm install' в каталоге Frontend.")
    if not environment_file.is_file():
        raise RuntimeError("Не найден .env. Создайте его из .env.example и заполните настройки.")
    for source_directory in ALL_SOURCE_DIRECTORIES:
        if not source_directory.is_dir():
            raise RuntimeError(f"Не найден каталог исходного кода: {source_directory}")

    return find_node_executable()


def find_node_executable() -> str:
    """
    Находит Node.js в PATH, стандартных менеджерах и локальном runtime Codex.
    """
    candidates: list[Path] = []
    path_candidate = shutil.which("node.exe") or shutil.which("node")
    if path_candidate is not None:
        candidates.append(Path(path_candidate))

    _append_environment_node_candidate(candidates, "NODE_HOME")
    _append_environment_node_candidate(candidates, "NVM_SYMLINK")

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "nodejs" / "node.exe")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "nodejs" / "node.exe")

    user_home = Path.home()
    candidates.extend(
        (
            user_home / ".volta" / "bin" / "node.exe",
            user_home / "scoop" / "apps" / "nodejs" / "current" / "node.exe",
            user_home
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "bin"
            / "node.exe",
        )
    )

    checked_paths: set[Path] = set()
    outdated_versions: list[str] = []
    for candidate in candidates:
        normalized_candidate = candidate.resolve(strict=False)
        if normalized_candidate in checked_paths or not normalized_candidate.is_file():
            continue
        checked_paths.add(normalized_candidate)
        version = read_node_version(normalized_candidate)
        if version is None:
            continue
        if version >= (22, 12, 0):
            return str(normalized_candidate)
        outdated_versions.append(f"{normalized_candidate} (версия {version})")

    if outdated_versions:
        found_versions = "; ".join(outdated_versions)
        raise RuntimeError(
            f"Нужен Node.js >=22.12. Найдены устаревшие версии: {found_versions}"
        )
    raise RuntimeError("Не найден Node.js >=22.12. Установите Node.js либо задайте NODE_HOME.")


def _append_environment_node_candidate(
    candidates: list[Path],
    variable_name: str,
) -> None:
    """
    Добавляет node.exe из переменной, содержащей каталог или полный путь.
    """
    raw_value = os.environ.get(variable_name)
    if not raw_value:
        return
    configured_path = Path(raw_value)
    candidates.append(
        configured_path
        if configured_path.name.lower() == "node.exe"
        else configured_path / "node.exe"
    )


def read_node_version(node_executable: Path) -> tuple[int, int, int] | None:
    """
    Возвращает версию запускаемого Node.js без shell-посредника.
    """
    try:
        completed = subprocess.run(
            [str(node_executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", completed.stdout.strip())
    return tuple(map(int, match.groups())) if match is not None else None


def build_python_environment(source_directories: tuple[Path, ...]) -> dict[str, str]:
    """
    Создаёт окружение дочернего Python-процесса с отдельным PYTHONPATH.
    """
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(directory) for directory in source_directories)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def start_managed_process(
    *,
    name: str,
    command: list[str],
    working_directory: Path,
    environment: dict[str, str],
    process_job: WindowsProcessJob | PosixProcessJob,
) -> ManagedProcess:
    """
    Запускает компонент без shell и регистрирует его дерево в Job Object.
    """
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        env=environment,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    try:
        process_job.assign(process)
    except Exception:
        process.kill()
        process.wait(timeout=5)
        raise
    return ManagedProcess(name=name, process=process)


def wait_until_ready(
    managed_process: ManagedProcess,
    url: str,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """
    Ожидает HTTP 200 или раннее завершение проверяемого компонента.
    """
    deadline = time.monotonic() + timeout_seconds
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        exit_code = managed_process.process.poll()
        if exit_code is not None:
            raise RuntimeError(f"Компонент {managed_process.name} завершился с кодом {exit_code}.")
        try:
            with opener.open(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(
        f"Компонент {managed_process.name} не стал готов за {timeout_seconds:g} секунд."
    )


def monitor_processes(processes: list[ManagedProcess]) -> None:
    """
    Работает до Ctrl+C или неожиданного завершения любого компонента.
    """
    while True:
        for managed_process in processes:
            exit_code = managed_process.process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Компонент {managed_process.name} завершился с кодом {exit_code}."
                )
        time.sleep(0.5)


def parse_arguments() -> argparse.Namespace:
    """
    Принимает файл окружения и локальные порты без передачи секретов через CLI.
    """
    parser = argparse.ArgumentParser(description="Локальный стенд ML + Backend + Frontend")
    parser.add_argument("--env-file", type=Path, default=ENVIRONMENT_FILE)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--backend-port", type=int, default=8001)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--check", action="store_true", help="Проверить настройки без запуска")
    parser.add_argument("--health-only", action="store_true", help="Разрешить запуск без LLM")
    arguments = parser.parse_args()
    ports = (arguments.api_port, arguments.backend_port, arguments.frontend_port)
    if len(set(ports)) != 3 or any(not 1 <= port <= 65535 for port in ports):
        parser.error("Требуются три разных порта от 1 до 65535")
    arguments.env_file = arguments.env_file.resolve()
    return arguments


def check_available_ports(ports: tuple[int, ...]) -> None:
    """
    Обнаруживает занятые порты до запуска и не останавливает чужие процессы.
    """
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            listener = socket.socket()
            sockets.append(listener)
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                raise RuntimeError(f"Локальный порт {port} недоступен") from None
    finally:
        for listener in sockets:
            listener.close()


def run_local_stack(arguments: argparse.Namespace) -> None:
    """
    Последовательно запускает и проверяет API, Backend и Frontend.
    """
    node_executable = validate_environment(arguments.env_file)
    python_environment = build_python_environment(ALL_SOURCE_DIRECTORIES)
    python_environment["QUESTIONNAIRE_ENV_FILE"] = str(arguments.env_file)
    python_environment["ML_TURN_URL"] = (
        f"http://127.0.0.1:{arguments.api_port}/v1/turns/process"
    )
    check_command = [str(PYTHON_EXECUTABLE), str(PROJECT_ROOT / "scripts" / "check_config.py")]
    if arguments.health_only:
        check_command.append("--health-only")
    result = subprocess.run(check_command, cwd=PROJECT_ROOT, env=python_environment, check=False)
    if result.returncode:
        raise RuntimeError("Конфигурация стенда не прошла проверку")
    check_available_ports((arguments.api_port, arguments.backend_port, arguments.frontend_port))
    if arguments.check:
        print("Зависимости, конфигурация и порты проверены.", flush=True)
        return
    frontend_environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("LLM_", "SERVICE_AUTH_", "QUESTIONNAIRE_"))
    }
    frontend_environment["VITE_BACKEND_TARGET"] = f"http://127.0.0.1:{arguments.backend_port}"
    process_job = WindowsProcessJob() if os.name == "nt" else PosixProcessJob()
    processes: list[ManagedProcess] = []

    try:
        api_process = start_managed_process(
            name="ML API",
            command=[
                str(PYTHON_EXECUTABLE),
                "-m",
                "uvicorn",
                "questionnaire_api.main:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(arguments.api_port),
                "--workers",
                "1",
            ],
            working_directory=PROJECT_ROOT,
            environment=python_environment,
            process_job=process_job,
        )
        processes.append(api_process)
        wait_until_ready(api_process, f"http://127.0.0.1:{arguments.api_port}/health/ready")

        backend_process = start_managed_process(
            name="Backend",
            command=[
                str(PYTHON_EXECUTABLE),
                "-m",
                "uvicorn",
                "questionnaire_backend.main:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(arguments.backend_port),
                "--workers",
                "1",
            ],
            working_directory=PROJECT_ROOT,
            environment=python_environment,
            process_job=process_job,
        )
        processes.append(backend_process)
        wait_until_ready(backend_process, f"http://127.0.0.1:{arguments.backend_port}/health/ready")

        frontend_process = start_managed_process(
            name="Frontend",
            command=[
                node_executable,
                str(VITE_SCRIPT),
                "--host",
                "127.0.0.1",
                "--port",
                str(arguments.frontend_port),
                "--strictPort",
            ],
            working_directory=PROJECT_ROOT / "Frontend",
            environment=frontend_environment,
            process_job=process_job,
        )
        processes.append(frontend_process)
        wait_until_ready(frontend_process, f"http://127.0.0.1:{arguments.frontend_port}/")

        print("", flush=True)
        print("Все компоненты готовы.", flush=True)
        print(
            f"Интерфейс: http://127.0.0.1:{arguments.frontend_port} "
            f"(PID {frontend_process.process.pid})",
            flush=True,
        )
        print(
            f"ML API: http://127.0.0.1:{arguments.api_port} (PID {api_process.process.pid})",
            flush=True,
        )
        print(
            f"Backend: http://127.0.0.1:{arguments.backend_port} "
            f"(PID {backend_process.process.pid})",
            flush=True,
        )
        print("Для остановки нажмите Ctrl+C или кнопку Stop в PyCharm.", flush=True)
        print("", flush=True)
        monitor_processes(processes)
    finally:
        process_job.close()
        for managed_process in processes:
            try:
                managed_process.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                managed_process.process.kill()
        print("Локальный стенд остановлен.", flush=True)


def main() -> int:
    """
    Возвращает код завершения, удобный для конфигурации запуска PyCharm.
    """
    arguments = parse_arguments()
    if os.name != "nt":
        signal.signal(signal.SIGTERM, handle_termination)
    try:
        run_local_stack(arguments)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Ошибка запуска: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


def handle_termination(signum: int, frame: object) -> None:
    """
    Переводит SIGTERM в штатную остановку с очисткой дочерних процессов.
    """
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
