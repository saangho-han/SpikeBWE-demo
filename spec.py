import os
import numpy as np   # ✅ 추가
import librosa
import librosa.display
import matplotlib.pyplot as plt

def save_spectrograms_from_folder(folder_path):
    # 폴더 내 모든 파일 탐색
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            
            # 오디오 로드
            y, sr = librosa.load(file_path, sr=None)
            
            # 스펙트로그램 계산 (dB scale)
            S = librosa.stft(y, n_fft=1024, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)  # ✅ np.abs 로 수정
            
            # 그림 생성
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', cmap='magma')
            plt.title(f"Spectrogram: {file_name}")
            plt.tight_layout()
            
            # 저장 경로 (wav 대신 png)
            save_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + ".png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved: {save_path}")


# 사용 예시
if __name__ == "__main__":
    folder = "C:\\Users\\ASUS\\Desktop\\test\\samples\\audio\\SX374\\"  # 여기에 음성 파일 폴더 경로 넣기
    save_spectrograms_from_folder(folder)