import subprocess, os, sys

PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_PINHOLE_FISHEYE"}
MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4,
    "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "SIMPLE_BROWN": 5, "BROWN": 7,
    "FISHEYE": 8, "SIMPLE_FISHEYE": 4,
    "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
}

def fix_one(target):
    print(f"=== {target} ===", flush=True)
    cameras_txt = f"{target}/cameras.txt"
    cameras_bin = f"{target}/cameras.bin"
    if not os.path.exists(cameras_txt):
        subprocess.run(
            ["colmap", "model_converter",
             "--input_path", target, "--output_path", target,
             "--output_type", "TXT"],
            check=True, cwd=target,
        )
    with open(cameras_txt) as f:
        lines = f.readlines()
    new_lines = []
    n = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            continue
        parts = line.strip().split()
        cam_id, model, w, h, *params = parts
        if model in PINHOLE_MODELS:
            new_lines.append(line)
            continue
        keep = MODEL_NUM_PARAMS.get(model, 4)
        if model == "SIMPLE_RADIAL":
            keep = 3
        new_model = "SIMPLE_PINHOLE" if model.startswith("SIMPLE_") else "PINHOLE"
        new_params = params[:keep]
        new_lines.append(" ".join([cam_id, new_model, w, h, *new_params]))
        n += 1
    if n > 0:
        with open(cameras_txt, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        subprocess.run(
            ["colmap", "model_converter",
             "--input_path", target, "--output_path", target,
             "--output_type", "BIN"],
            check=True, cwd=target,
        )
        print(f"  fixed {n} cameras, regenerated .bin", flush=True)
    else:
        print(f"  no fix needed (all PINHOLE family already)", flush=True)
    return n

job_id = "20260911044811-54-55-a47b"
job_dir = f"/home/jobs/{job_id}"
for t in [f"{job_dir}/distort_free/sparse/0", f"{job_dir}/sparse/0"]:
    fix_one(t)
print("DONE", flush=True)
