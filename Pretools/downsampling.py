import os
import argparse
from PIL import Image

# ===================== 命令行参数配置 =====================
parser = argparse.ArgumentParser(description="批量对图片进行下采样")
parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹路径")
parser.add_argument("--output_dir", type=str, required=True, help="输出图片文件夹路径")
parser.add_argument("--scale", type=int, default=8, help="下采样倍数，例如 8 代表缩小为 1/8")
args = parser.parse_args()
# ==========================================================

# 从参数获取路径和缩放倍数
input_folder = args.input_dir
output_folder = args.output_dir
scale = args.scale

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有图片文件
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 循环处理
for image_file in image_files:
    try:
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)
        width, height = image.size

        # 动态计算新尺寸
        new_width = width // scale
        new_height = height // scale

        downscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        output_path = os.path.join(output_folder, image_file)
        downscaled_image.save(output_path)

        print(f"完成: {image_file} -> {new_width}x{new_height}")
    except Exception as e:
        print(f"跳过 {image_file}, 错误: {e}")

print("所有图片下采样完成！")