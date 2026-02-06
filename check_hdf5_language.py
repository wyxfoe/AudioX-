"""
检查 RoboTwin HDF5 文件中的语言描述字段
在你的机器上运行此脚本
"""
import h5py
import sys

def check_hdf5_for_language(hdf5_path):
    """检查 HDF5 文件中是否有语言描述"""
    print(f"检查文件: {hdf5_path}\n")

    with h5py.File(hdf5_path, 'r') as f:
        # 检查所有 attributes
        print("=== 文件级 Attributes ===")
        for key in f.attrs.keys():
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            print(f"  {key}: {val}")

        # 递归检查所有组的 attributes
        def check_group(group, prefix=""):
            for key in group.keys():
                item = group[key]
                full_path = f"{prefix}/{key}" if prefix else key

                # 检查 attributes
                if len(item.attrs) > 0:
                    print(f"\n=== {full_path} Attributes ===")
                    for attr_key in item.attrs.keys():
                        val = item.attrs[attr_key]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        print(f"  {attr_key}: {val}")

                # 递归检查子组
                if isinstance(item, h5py.Group):
                    check_group(item, full_path)

                # 检查是否有 string 类型的 dataset（可能是语言描述）
                elif isinstance(item, h5py.Dataset):
                    if item.dtype.kind == 'S' or item.dtype.kind == 'O':  # string types
                        if 'rgb' not in full_path and 'image' not in full_path.lower():
                            print(f"\n=== {full_path} (string dataset) ===")
                            try:
                                data = item[()]
                                if isinstance(data, bytes):
                                    print(f"  内容: {data.decode('utf-8')[:200]}")
                                elif isinstance(data, str):
                                    print(f"  内容: {data[:200]}")
                            except:
                                print(f"  (无法读取)")

        check_group(f)

        # 特别检查常见的语言描述字段名
        print("\n=== 检查常见语言字段 ===")
        common_keys = [
            'language', 'language_instruction', 'instruction', 'task',
            'task_description', 'description', 'text', 'prompt',
            'attrs/language', 'attrs/instruction'
        ]
        for key in common_keys:
            if key in f:
                print(f"  找到 '{key}': {f[key][()]}")
            elif key in f.attrs:
                print(f"  找到 attrs['{key}']: {f.attrs[key]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_hdf5_for_language(sys.argv[1])
    else:
        print("用法: python check_hdf5_language.py /path/to/episode.hdf5")
        print("\n示例:")
        print("  python check_hdf5_language.py /home/wyx/Beijingwork/RoboTwin/data/beat_block_hammer/demo_clean_500/data/episode1.hdf5")
