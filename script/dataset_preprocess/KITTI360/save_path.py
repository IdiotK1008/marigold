import os

# 定义根目录
root_dir = './data/KITTI360/data_2d_raw'
output_file = 'filename_list_eval_full.txt'

# 打开一个文件用于写入
with open(output_file, 'w') as f:
    # 遍历根目录下的所有 scene 文件夹
    for scene in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_path):
            continue  # 跳过非目录文件
        scene_id = scene.split('_')[-2]
        if int(scene_id) != 10:
            continue  # 跳过 scene 9 及之后的文件夹

        # 只处理 image_2 和 image_3
        for image_folder in ['image_02', 'image_03']:
            image_path = os.path.join(scene_path, image_folder)
            if not os.path.exists(image_path):
                continue  # 跳过不存在的文件夹

            # 获取 rgb 和 depth 文件夹路径
            rgb_path = os.path.join(image_path, 'data_rgb')
            depth_path = os.path.join(image_path, 'distancemap')

            # 检查 rgb 和 depth 文件夹是否存在
            if not os.path.exists(rgb_path):
                print(rgb_path, "not exist")
                continue  # 跳过不存在的文件夹
            if not os.path.exists(depth_path):
                print(depth_path, "not exist")
                continue  # 跳过不存在的文件夹

            # 获取 rgb 文件夹中的所有文件
            rgb_files = sorted(os.listdir(rgb_path))
            depth_files = sorted(os.listdir(depth_path))

            # 确保 rgb 和 depth 中的文件一一对应
            for rgb_file, depth_file in zip(rgb_files, depth_files):
                # 构建相对路径
                rgb_relative_path = os.path.relpath(os.path.join(rgb_path, rgb_file), root_dir)
                depth_relative_path = os.path.relpath(os.path.join(depth_path, depth_file), root_dir)

                # 将 rgb 和 depth 的路径写入同一行
                f.write(f"{rgb_relative_path} {depth_relative_path}\n")

print(f"所有符合条件的文件路径已保存到 {output_file} 文件中。")