"""Track 0.1 gate: verify the GPU stack actually works on Blackwell (sm_120).

Older cu121 wheels install cleanly on this machine and then fail at runtime with
"no kernel image is available for execution on the device". The arch-list check
below is the only reliable way to catch that before it wastes an experiment run.
"""

import sys

import torch


def main() -> int:
    print(f"torch            : {torch.__version__}")
    print(f"cuda build       : {torch.version.cuda}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    arch_list = torch.cuda.get_arch_list()
    print(f"arch_list        : {arch_list}")

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"device           : {name}")
    print(f"capability       : {cap}")
    print(f"total VRAM (MiB) : {total:.0f}")

    ok = True
    if cap != (12, 0):
        print(f"WARN: expected capability (12, 0), got {cap}")
    if "sm_120" not in arch_list:
        print("FAIL: torch was not built with sm_120 kernels -- reinstall from the cu128 index")
        ok = False

    # Real kernel launch: this is what actually fails on a mismatched build.
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        c = (a @ b).float()
        torch.cuda.synchronize()
        print(f"fp16 matmul      : ok (mean={float(c.mean()):+.5f})")
        print(f"peak alloc (MiB) : {torch.cuda.max_memory_allocated() / 2**20:.1f}")
    except Exception as exc:  # noqa: BLE001 - we want the raw failure text
        print(f"FAIL: matmul raised {type(exc).__name__}: {exc}")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
