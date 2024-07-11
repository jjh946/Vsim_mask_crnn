import os

# 테스트를 위한 이미지와 어노테이션 디렉토리 경로 설정
image_dir = '../images'
bounding_box_dir = '../../Saved/1bea28d9bb1d3d9acd7b5e420719f05a/labelingData/Car/N01S01M01/Design0008/outputJson/boundingbox2d'
polygon_dir = '../../Saved/1bea28d9bb1d3d9acd7b5e420719f05a/labelingData/Car/N01S01M01/Design0008/outputJson/polygon'

# 경로에 있는 파일 목록 출력
print("Checking paths and files...")

def check_path(path):
    if os.path.exists(path):
        print(f"Path exists: {path}")
        files = os.listdir(path)
        if files:
            print(f"Files in {path}: {files[:5]} ...")  # 처음 5개의 파일만 출력
        else:
            print(f"No files found in {path}")
    else:
        print(f"Path does not exist: {path}")

check_path(image_dir)
check_path(bounding_box_dir)
check_path(polygon_dir)
