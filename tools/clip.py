import os
import subprocess

INPUT_VIDEO = "/mnt/hdd/qiuxia/pyr/LF3R/datasets/lf3r_failure_rollouts/v1/failrecovery/20260929_233446_usb_0929_mixedfail_0023/videos/wrist_right.mp4"  # 替换为你的视频路径

events = [
    {"type": "dropped_object", "start": 185, "end": 204},
    {"type": "wrong_object", "start": 415, "end": 444},
    {"type": "wrong_object", "start": 513, "end": 549},
    {"type": "dropped_object", "start": 560, "end": 575},
]
for idx, e in enumerate(events, 1):
    out_name = f"clip_{idx}_{e['type']}_{e['start']}_{e['end']}.mp4"
    # 直接根据帧号范围过滤 (精确按帧抽取)
    vf = f"select='between(n\\,{e['start']}\\,{e['end']})',setpts=PTS-STARTPTS"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", INPUT_VIDEO,
        "-vf", vf,
        "-an",  # 如果不需要音频保留无声，加速处理
        out_name
    ]
    
    print(f"正在裁剪: {out_name} (帧 {e['start']} -> {e['end']})")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("完成！")