# -*- coding: utf-8 -*-
"""
Sincroniza o Data Lake local (`dados/parquet_nacional/` + `dados/historico_cargas.json`)
com uma GitHub Release dedicada (`lakehouse-data`), evitando commits de centenas de MB no Git
enquanto permite execucao automatica stateless no GitHub Actions.

Uso:
  python scripts/sync_lakehouse_release.py download [--tag lakehouse-data]
  python scripts/sync_lakehouse_release.py upload [--anos 2026] [--dry-run]
"""

import argparse
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = BASE_DIR / "dados"
PARQUET_DIR = DADOS_DIR / "parquet_nacional"
HISTORICO_PATH = DADOS_DIR / "historico_cargas.json"
DEFAULT_TAG = "lakehouse-data"


def _inferir_repo() -> str | None:
    env_repo = os.environ.get("GITHUB_REPOSITORY")
    if env_repo:
        return env_repo.strip()
    try:
        out = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=BASE_DIR, text=True).strip()
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["gh"] + args
    return subprocess.run(cmd, cwd=BASE_DIR, text=True, capture_output=True, check=check)


def cmd_upload(tag: str, anos: list[int] | None, dry_run: bool = False):
    if not PARQUET_DIR.exists():
        print("Nenhum diretório parquet_nacional encontrado para empacotar.")
        return

    anos_dirs = sorted(d for d in PARQUET_DIR.glob("ano=*") if d.is_dir())
    if anos:
        alvo = {f"ano={a}" for a in anos}
        anos_dirs = [d for d in anos_dirs if d.name in alvo]

    if not anos_dirs:
        print("Nenhum ano correspondente encontrado em dados/parquet_nacional/.")
        return

    repo = _inferir_repo()
    print(f"Preparando empacotamento Lakehouse -> tag '{tag}' (repo={repo or 'auto'})", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        assets: list[Path] = []
        for ano_dir in anos_dirs:
            ano_num = ano_dir.name.split("=")[-1]
            tar_path = tmp_path / f"parquet_{ano_num}.tar"
            with tarfile.open(tar_path, "w") as tf:
                tf.add(ano_dir, arcname=ano_dir.name)
            mb = tar_path.stat().st_size / (1024 * 1024)
            n_parts = sum(1 for _ in ano_dir.rglob("*.parquet"))
            print(f"  ✓ {tar_path.name}: {mb:.2f} MB ({n_parts} partições)", flush=True)
            assets.append(tar_path)

        if HISTORICO_PATH.exists():
            hist_copy = tmp_path / "historico_cargas.json"
            shutil.copy2(HISTORICO_PATH, hist_copy)
            assets.append(hist_copy)
            print(f"  ✓ {hist_copy.name}: {hist_copy.stat().st_size / 1024:.1f} KB", flush=True)

        if dry_run:
            print("[DRY-RUN] Empacotamento validado com sucesso (upload ignorado).", flush=True)
            return

        repo_flags = ["--repo", repo] if repo else []
        view = _run_gh(["release", "view", tag] + repo_flags, check=False)
        if view.returncode != 0:
            print(f"Criando Release '{tag}' no GitHub...", flush=True)
            _run_gh(
                ["release", "create", tag, "--title", "SIH/SUS Data Lakehouse (Parquets)", "--notes", "Armazenamento incremental Parquet Hive para pipeline MLOps automático."]
                + repo_flags,
                check=True,
            )

        print(f"Enviando {len(assets)} asset(s) para Release '{tag}'...", flush=True)
        _run_gh(["release", "upload", tag, "--clobber"] + [str(a) for a in assets] + repo_flags, check=True)
        print("✓ Upload Lakehouse na GitHub Release concluído!", flush=True)


def cmd_download(tag: str, anos: list[int] | None, strict: bool = False):
    repo = _inferir_repo()
    repo_flags = ["--repo", repo] if repo else []
    view = _run_gh(["release", "view", tag] + repo_flags, check=False)
    if view.returncode != 0:
        msg = f"Release '{tag}' ainda não existe no GitHub (primeira execução ou publicação inicial pendente)."
        if strict:
            raise RuntimeError(msg)
        print(f"[AVISO] {msg} Seguindo com o que houver em disco/FTP.", flush=True)
        return

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        patterns = ["-p", "historico_cargas.json"]
        if anos:
            for a in anos:
                patterns += ["-p", f"parquet_{a}.tar"]
        else:
            patterns += ["-p", "parquet_*.tar"]

        print(f"Baixando Lakehouse da Release '{tag}'...", flush=True)
        _run_gh(["release", "download", tag, "--dir", str(tmp_path)] + patterns + repo_flags, check=False)

        hist_dl = tmp_path / "historico_cargas.json"
        if hist_dl.exists():
            shutil.copy2(hist_dl, HISTORICO_PATH)
            print("  ✓ historico_cargas.json restaurado.", flush=True)

        tars = sorted(tmp_path.glob("parquet_*.tar"))
        for tar_path in tars:
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(PARQUET_DIR)
            print(f"  ✓ {tar_path.name} extraído em {PARQUET_DIR}.", flush=True)

        if not tars and strict:
            raise RuntimeError(f"Nenhum arquivo parquet_*.tar baixado da Release '{tag}'.")


def main():
    parser = argparse.ArgumentParser(description="Sync do Data Lake SIH/SUS via GitHub Releases")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="Empacotar anos Parquet e enviar para GitHub Release")
    p_up.add_argument("--tag", default=DEFAULT_TAG)
    p_up.add_argument("--anos", nargs="+", type=int, default=None, help="Filtrar anos específicos")
    p_up.add_argument("--dry-run", action="store_true")

    p_dn = sub.add_parser("download", help="Baixar e extrair Data Lake de uma GitHub Release")
    p_dn.add_argument("--tag", default=DEFAULT_TAG)
    p_dn.add_argument("--anos", nargs="+", type=int, default=None)
    p_dn.add_argument("--strict", action="store_true")

    args = parser.parse_args()
    if args.cmd == "upload":
        cmd_upload(args.tag, args.anos, dry_run=args.dry_run)
    elif args.cmd == "download":
        cmd_download(args.tag, args.anos, strict=args.strict)


if __name__ == "__main__":
    main()
