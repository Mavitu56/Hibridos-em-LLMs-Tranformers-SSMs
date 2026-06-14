"""
setup_env.py — Instalação dos kernels Mamba-2 por WHEEL pré-compilada, com
fallback gracioso para o backend PyTorch puro (transformers.Mamba2).

CONTEXTO (Tarefa 1 da spec):
    O erro `undefined symbol: _ZN3c10...` em `selective_scan_cuda.so` é uma
    incompatibilidade de ABI C++ — NÃO ausência de wheel. `pip install
    --no-build-isolation mamba-ssm` falha porque manda COMPILAR do zero contra
    uma ABI que não casa. A solução é instalar a wheel pré-compilada EXATA que
    corresponde a (torch, cuda, cxx11abi, python), pulando a compilação.

    Formato das wheels nos releases:
        mamba_ssm-<ver>+cu<XX>torch<Y.Z>cxx11abi<TRUE|FALSE>-cp<PY>-cp<PY>-linux_x86_64.whl
        causal_conv1d-<ver>+cu<XX>torch<Y.Z>cxx11abi<TRUE|FALSE>-cp<PY>-cp<PY>-linux_x86_64.whl

    Releases:
        mamba-ssm:     https://github.com/state-spaces/mamba/releases
        causal-conv1d: https://github.com/Dao-AILab/causal-conv1d/releases

    A flag cxx11abi é a que mais quebra: é DERIVADA de torch._C._GLIBCXX_USE_CXX11_ABI,
    nunca chutada.

Uso (no Colab, no MESMO processo — não usar !python em subprocesso):
    import setup_env
    backend = setup_env.setup()      # instala wheels OU cai p/ "torch"
    # backend in {"kernels", "torch"}; também exportado em os.environ["MAMBA_BACKEND"]
"""

import importlib
import os
import subprocess
import sys


# Versões alvo dos releases (verificadas em 2026-06-12). As wheels destes
# releases cobrem cu11/cu12/cu13 × torch 2.6–2.10 × cp310–cp313 × cxx11abi
# TRUE/FALSE. Ajuste se um release mais novo casar melhor.
MAMBA_SSM_VERSION = "2.3.2.post1"
CAUSAL_CONV1D_VERSION = "1.6.2.post1"

# O Colab pode rodar um torch MAIS NOVO que a última wheel publicada (observado
# em 2026-06-12: torch 2.11.0+cu128 no Colab; wheels só até torch2.10 → 404).
# Extensões compiladas contra um libtorch minor anterior frequentemente importam
# no seguinte; tentamos minors anteriores em ordem decrescente e deixamos o
# SMOKE TEST decidir — se o import/forward falhar, cai para o backend torch.
TORCH_FALLBACK_MINORS = ["2.10", "2.9", "2.8", "2.7", "2.6"]

MAMBA_RELEASE_BASE = (
    "https://github.com/state-spaces/mamba/releases/download/"
    f"v{MAMBA_SSM_VERSION}"
)
CAUSAL_RELEASE_BASE = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/"
    f"v{CAUSAL_CONV1D_VERSION}"
)


# ---------------------------------------------------------------------------
# Detecção de ambiente
# ---------------------------------------------------------------------------

def detect_env() -> dict:
    """
    Lê o ambiente de runtime e devolve os campos que compõem o nome da wheel.

    O campo `abi` (cxx11abi) é o erro #1 de instalação: derivamos de
    torch._C._GLIBCXX_USE_CXX11_ABI, jamais chutamos.
    """
    import torch

    torch_ver = torch.__version__.split('+')[0]            # ex. "2.6.0"
    cuda_ver = torch.version.cuda                          # ex. "12.4" ou None (CPU build)
    abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)            # True/False
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"  # ex. "cp311"

    # As wheels usam major.minor do torch no nome (ex. torch2.6), e cuXX = major da CUDA.
    torch_mm = ".".join(torch_ver.split(".")[:2])          # "2.6"
    cu_short = f"cu{cuda_ver.split('.')[0]}" if cuda_ver else None  # "cu12"

    env = {
        "torch_ver": torch_ver,
        "torch_mm": torch_mm,
        "cuda_ver": cuda_ver,
        "cu_short": cu_short,
        "abi": abi,
        "abi_str": "TRUE" if abi else "FALSE",
        "py": py,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
    }
    return env


def _wheel_name(pkg: str, version: str, env: dict) -> str:
    """Monta o nome do arquivo .whl para um pacote dado o ambiente."""
    cu = env["cu_short"]
    torch_mm = env["torch_mm"]
    abi = env["abi_str"]
    py = env["py"]
    return (
        f"{pkg}-{version}+{cu}torch{torch_mm}cxx11abi{abi}-"
        f"{py}-{py}-linux_x86_64.whl"
    )


def build_wheel_urls(env: dict) -> dict:
    """Devolve as URLs candidatas para causal-conv1d e mamba-ssm."""
    causal_file = _wheel_name("causal_conv1d", CAUSAL_CONV1D_VERSION, env)
    mamba_file = _wheel_name("mamba_ssm", MAMBA_SSM_VERSION, env)
    return {
        "causal_conv1d": f"{CAUSAL_RELEASE_BASE}/{causal_file}",
        "mamba_ssm": f"{MAMBA_RELEASE_BASE}/{mamba_file}",
    }


# ---------------------------------------------------------------------------
# Instalação
# ---------------------------------------------------------------------------

def _pip_install(target: str) -> bool:
    """pip install <target>; True se exit 0. Loga a saída."""
    print(f"  pip install {target}")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", target],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Mostra só o fim do stderr (o erro real de ABI/404 fica aqui).
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"  [falhou] {target}\n{tail}")
        return False
    return True


def install_kernels(env: dict, torch_mm: str = None) -> bool:
    """
    Instala causal-conv1d PRIMEIRO, depois mamba-ssm (ordem importa: mamba-ssm
    linka contra causal-conv1d). Devolve True se ambos instalarem.

    torch_mm: sobrepõe o minor do torch no nome da wheel (fallback p/ quando o
    torch do runtime é mais novo que a última wheel publicada).
    """
    if not env["cuda_available"]:
        print("  CUDA indisponível — kernels Mamba exigem GPU. Pulando para fallback.")
        return False
    if env["cu_short"] is None:
        print("  torch sem build CUDA — kernels não aplicáveis. Pulando para fallback.")
        return False

    if torch_mm is not None and torch_mm != env["torch_mm"]:
        env = {**env, "torch_mm": torch_mm}
        print(f"  [fallback] tentando wheels compiladas p/ torch {torch_mm} "
              f"(runtime tem {env['torch_ver']}; o smoke test decide).")

    urls = build_wheel_urls(env)
    print(f"  causal-conv1d: {urls['causal_conv1d']}")
    print(f"  mamba-ssm:     {urls['mamba_ssm']}")

    if not _pip_install(urls["causal_conv1d"]):
        return False
    if not _pip_install(urls["mamba_ssm"]):
        return False
    return True


def _torch_minor_candidates(env: dict) -> list:
    """
    Minors de torch a tentar no nome da wheel: o do runtime primeiro, depois os
    anteriores (nunca um minor mais novo que o runtime — ABI futura não linka).
    """
    def as_tuple(mm: str):
        major, minor = mm.split(".")[:2]
        return (int(major), int(minor))

    current = env["torch_mm"]
    cands = [current]
    for mm in TORCH_FALLBACK_MINORS:
        if mm != current and as_tuple(mm) < as_tuple(current):
            cands.append(mm)
    return cands


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_kernels() -> bool:
    """Importa Mamba2 dos kernels e faz um forward em CUDA. True se passar."""
    try:
        import torch
        # invalida caches de import caso a instalação tenha ocorrido neste processo
        importlib.invalidate_caches()
        from mamba_ssm import Mamba2
    except Exception as e:
        print(f"  smoke import falhou: {e}")
        return False

    try:
        m = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2).cuda()
        x = torch.randn(2, 64, 128, device="cuda")
        y = m(x)
        ok = y.shape == x.shape and torch.isfinite(y).all()
        print(f"  smoke forward: out={tuple(y.shape)} finito={bool(torch.isfinite(y).all())}")
        return bool(ok)
    except Exception as e:
        print(f"  smoke forward falhou: {e}")
        return False


def smoke_torch_backend() -> bool:
    """Verifica que o fallback PyTorch puro (transformers.Mamba2) está disponível."""
    try:
        importlib.invalidate_caches()
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer  # noqa: F401
        print("  transformers.Mamba2Mixer disponível (backend torch puro).")
        return True
    except Exception as e:
        print(f"  fallback torch indisponível: {e}")
        print("  Instale 'transformers>=4.40' e 'einops'.")
        return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def setup(timebox_note: bool = True) -> str:
    """
    Tenta instalar os kernels por wheel; cai para o backend torch puro se não casar.
    Define os.environ["MAMBA_BACKEND"] e o retorna ∈ {"kernels", "torch"}.
    """
    print("=" * 60)
    print("setup_env — backend Mamba-2")
    print("=" * 60)

    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "PyTorch não encontrado. Instale torch antes de rodar setup_env "
            "(no Colab já vem instalado)."
        )

    env = detect_env()
    print("Ambiente detectado:")
    for k in ("torch_ver", "cuda_ver", "abi", "py", "gpu_name", "bf16_supported"):
        print(f"  {k:16s}= {env[k]}")
    print()

    backend = "torch"  # default conservador

    print("[1/2] Tentando instalar kernels CUDA por wheel pré-compilada...")
    kernels_ok = False
    for torch_mm in _torch_minor_candidates(env):
        if install_kernels(env, torch_mm) and smoke_kernels():
            kernels_ok = True
            break

    if kernels_ok:
        backend = "kernels"
        print("✓ Backend KERNELS ativo (fast path do Mamba-2).")
    else:
        if timebox_note:
            print(
                "  Wheel não casou ou smoke falhou. NÃO vamos compilar do zero "
                "(timebox da spec). Caindo para o backend torch puro."
            )
        print("\n[2/2] Validando fallback PyTorch puro (transformers.Mamba2)...")
        if smoke_torch_backend():
            backend = "torch"
            print("✓ Backend TORCH ativo (SSD puro PyTorch, idêntico a Dao & Gu 2024).")
        else:
            raise RuntimeError(
                "Nenhum backend Mamba disponível: kernels falharam E o fallback "
                "torch (transformers+einops) não está instalado."
            )

    os.environ["MAMBA_BACKEND"] = backend
    print(f"\nMAMBA_BACKEND = {backend}")
    print("=" * 60)
    return backend


if __name__ == "__main__":
    setup()
