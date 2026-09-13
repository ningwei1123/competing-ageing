"""探测 Severson struct .mat 格式并解析 summary 字段 (逐周期 Qd/IR/T)."""
import os, sys
import numpy as np

RAW = "data/severson_raw"


def detect_format(path):
    with open(path, "rb") as f:
        head = f.read(128)
    s = head[:64]
    if b"MATLAB 7.3" in head:
        return "v7.3"
    if b"MATLAB 5.0" in head:
        return "v7"
    return "unknown: " + repr(s[:40])


def load_summary_scipy(path):
    """v7: scipy.io.loadmat, 返回 (qd_list, ir_list, t_list, cycle_life_list)."""
    from scipy.io import loadmat
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    # 找 struct 变量 (通常 'batch')
    key = [k for k in data if not k.startswith("__") and
           hasattr(data[k], "dtype") and data[k].dtype.names]
    print("  variables:", [k for k in data if not k.startswith("__")])
    return data


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if p is None:
        cands = [f for f in os.listdir(RAW) if f.endswith(".mat")
                 and not f.endswith(".part")]
        for c in sorted(cands):
            fp = os.path.join(RAW, c)
            print(f"{c}: {detect_format(fp)}  ({os.path.getsize(fp)/1e6:.0f} MB)")
        return
    print(f"format: {detect_format(p)}")
    if detect_format(p) == "v7":
        load_summary_scipy(p)


if __name__ == "__main__":
    main()
