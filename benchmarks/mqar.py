"""
benchmarks/mqar.py — SHIM de compatibilidade.

A implementação do MQAR migrou para eval/mqar.py (Tarefa 7 da spec, pasta eval/).
Este arquivo permanece apenas para não quebrar imports/notebooks antigos que
faziam `from benchmarks.mqar import ...`. Prefira `eval.mqar` em código novo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mqar import (  # noqa: F401,E402
    generate_mqar_examples,
    evaluate_mqar,
    selftest,
    main,
)

if __name__ == "__main__":
    main()
