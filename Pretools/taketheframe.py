import cv2
import os
import argparse


def auto_extract_frames(video_path, output_dir, time_interval=None, frame_interval=None):
    """
    自动提取视频帧
    :param video_path: BlueStatues.mp4
    :param output_dir: C:/Datasets/Gym
    :param time_interval: 1
    :param frame_interval: 30
    """

    # 检查参数有效性
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件 {video_path} 不存在")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if time_interval and frame_interval:
        raise ValueError("不能同时设置时间间隔和帧数间隔")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频文件")

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 计算间隔参数
    if time_interval:
        frame_interval = int(fps * time_interval)
    elif not frame_interval:
        frame_interval = 10  # 默认30帧取一次

    print(f"视频信息：{fps}FPS，总帧数：{total_frames}")
    print(f"提取间隔：{frame_interval}帧（约{frame_interval / fps:.1f}秒）")

    # 开始提取
    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 修改 #selectedCode 中的代码
        if count % frame_interval == 0:
            # 生成文件名：6 位 zero-padded 序号 .png(无损,适合 3DGS 输入)
            frame_name = f"{saved_count + 1:06d}.png"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"处理完成，共保存 {saved_count} 帧图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频自动取帧工具")
    parser.add_argument("--video_path",
                        default="/home/yinjian/5mgo.MP4")
    parser.add_argument("--output_dir",
                        default="/home/yinjian/images5mgo")
    parser.add_argument("-t", "--time_interval",  type=float)
    parser.add_argument("-f", "--frame_interval", default="45", type=int)

    args = parser.parse_args()

    auto_extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        time_interval=args.time_interval,
        frame_interval=args.frame_interval
    )