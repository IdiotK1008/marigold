import os
import shutil

def reorganize_files(target_dir):
    # Define subfolder names
    subfolders = ["depth", "filled", "rgb"]

    # Create main subfolders (depth, filled, rgb) in target directory
    for subfolder in subfolders:
        os.makedirs(os.path.join(target_dir, subfolder), exist_ok=True)

    # Iterate through each scene folder in the target directory
    for scene_name in os.listdir(target_dir):
        scene_path = os.path.join(target_dir, scene_name)
        
        if os.path.isdir(scene_path):  # Ensure it's a directory
            # Iterate through files in the scene folder
            for file_name in os.listdir(scene_path):
                if file_name.startswith("depth_"):
                    dest_folder = "depth"
                elif file_name.startswith("filled_"):
                    dest_folder = "filled"
                elif file_name.startswith("rgb_"):
                    dest_folder = "rgb"
                else:
                    continue  # Skip files that don't match

                # Define destination path
                dest_path = os.path.join(target_dir, dest_folder, scene_name)
                os.makedirs(dest_path, exist_ok=True)

                # Move file to the new destination
                shutil.move(
                    os.path.join(scene_path, file_name),
                    os.path.join(dest_path, file_name)
                )

            # Remove the empty scene folder
            if not os.listdir(scene_path):
                os.rmdir(scene_path)

if __name__ == "__main__":
    # reorganize_files('data/nyu')
    # print("Reorganization complete!")

    for folder in ['depth', 'filled', 'rgb']:
        for scene in os.listdir(os.path.join('data/nyu', folder)):
            if scene.split('.')[-1]!='png':
                for file in os.listdir(os.path.join('data/nyu', folder, scene)):
                    shutil.move(os.path.join('data/nyu', folder, scene, file), os.path.join('data/nyu', folder, file))

                os.rmdir(os.path.join('data/nyu', folder, scene))