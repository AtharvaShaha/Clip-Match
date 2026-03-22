import cv2
import os
import numpy as np

DATASET_PATH = "Dataset"
QUERY_VIDEO = "query/query_01.mp4"

FRAME_PATH = "temp_frames"
QUERY_FRAME_PATH = "temp_query_frames"

FPS_TARGET = 32
MATCH_THRESHOLD = 0.80


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_frames_single(video_path, save_folder):

    ensure_dir(save_folder)

    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(original_fps / FPS_TARGET), 1)

    count = 0
    saved = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:

            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            filename = f"{save_folder}/frame_{saved:04d}.jpg"
            cv2.imwrite(filename, frame)

            saved += 1

        count += 1

    cap.release()


def block_features(image):

    h, w = image.shape

    bh = h // 4
    bw = w // 4

    blocks = []

    for i in range(4):
        for j in range(4):

            block = image[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            blocks.append(np.mean(block))

    return np.array(blocks)


def ncc(a, b):

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))

    if denom == 0:
        return 0

    return np.sum(a*b) / denom


def compare_query_with_dataset():

    query_folder = os.listdir(QUERY_FRAME_PATH)[0]

    query_frames = sorted(
        os.listdir(os.path.join(QUERY_FRAME_PATH, query_folder))
    )

    best_video = None
    best_score = -1

    for dataset_video in os.listdir(FRAME_PATH):

        dataset_folder = os.path.join(FRAME_PATH, dataset_video)
        dataset_frames = sorted(os.listdir(dataset_folder))

        scores = []

        for i in range(min(len(query_frames), len(dataset_frames))):

            q_img = cv2.imread(
                os.path.join(QUERY_FRAME_PATH, query_folder, query_frames[i]), 0
            )

            d_img = cv2.imread(
                os.path.join(dataset_folder, dataset_frames[i]), 0
            )

            if q_img is None or d_img is None:
                continue

            q_feat = block_features(q_img)
            d_feat = block_features(d_img)

            score = ncc(q_feat, d_feat)

            scores.append(score)

            print(f"{dataset_video} | Frame {i} NCC = {score:.3f}")

        if len(scores) == 0:
            continue

        avg_score = np.mean(scores)

        print(f"\nAverage NCC with {dataset_video} = {avg_score:.3f}\n")

        if avg_score > best_score:
            best_score = avg_score
            best_video = dataset_video

    print("=================================")
    print("BEST MATCH:", best_video)
    print("SIMILARITY:", best_score)

    if best_score > MATCH_THRESHOLD:
        print("RESULT: VIDEO EXISTS IN DATASET ✅")
    else:
        print("RESULT: VIDEO NOT FOUND ❌")


print("Extracting dataset frames...")

for video in os.listdir(DATASET_PATH):

    extract_frames_single(
        os.path.join(DATASET_PATH, video),
        os.path.join(FRAME_PATH, video.split(".")[0])
    )

print("Extracting query frames...")

query_name = QUERY_VIDEO.split("/")[-1].split(".")[0]

extract_frames_single(
    QUERY_VIDEO,
    os.path.join(QUERY_FRAME_PATH, query_name)
)

print("Comparing frames using NCC...\n")

compare_query_with_dataset()