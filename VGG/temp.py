import os
import shutil

if __name__ == '__main__':
    # Set source and destination directories
    source_dir = '/home/washindeiru/studia/sem_8/ssn/projekt_v2/ResidualAttentionNetwork-pytorch/VGG/data_2016/Train_Lesion/benign'
    destination_dir = '/home/washindeiru/studia/sem_8/ssn/projekt_v2/ResidualAttentionNetwork-pytorch/VGG/data_2016/Train/benign'

    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Move .jpg files
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.jpg'):
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(destination_dir, filename)
            shutil.move(src_path, dest_path)
            print(f"Moved: {filename}")

    print("All .jpg files have been moved.")
