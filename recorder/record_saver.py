import os
from datetime import datetime
from pathlib import Path


def save_signin_record(signed_in: set, base_names: set, output_dir=None):
    if not signed_in:
        print("\nℹ️ 无人签到。")
        return None

    # 如果未指定输出目录，则默认为项目根目录下的 Signin-record
    if output_dir is None:
        # 获取当前文件所在目录的上两级：项目根目录
        current_dir = Path(__file__).parent.parent
        output_dir = current_dir / "Signin-record"

    # 创建目录
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"signin_record_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("课堂人脸签到记录\n")
        f.write("=" * 30 + "\n")
        for student in sorted(signed_in):
            f.write(f"{student}\n")

        unsigned = base_names - signed_in
        if unsigned:
            f.write("\n未签到的学生:\n")
            f.write("=" * 30 + "\n")
            for student in sorted(unsigned):
                f.write(f"{student}\n")

    print(f"\n✅ 签到记录已保存至: {filename.absolute()}")
    return str(filename)