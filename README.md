# Computer Vision Basics

- 컴퓨터 비전 교재 순서에 따라 이론을 정리하고 실습을 구현한 저장소
- 참고 교재: IT CookBook, 컴퓨터 비전과 딥러닝 (오일석, 한빛아카데미, 2023)
- 사용 라이브러리: Python, OpenCV, NumPy, PyQt5

## 보유 역량

| 역량 | 할 수 있는 일 | 근거 | 라이브러리 |
|---|---|---|---|
| 객체 인식 구현 | 영상에서 기준 이미지(표지판)를 찾아 종류를 판별하고 위치를 추정하는 흐름 구현 | [sift_sign_recognition](./sift_sign_recognition) | OpenCV, NumPy |
| 특징점 기반 매칭 | SIFT 특징점 추출, FLANN 매칭, 거리 비율 검사와 RANSAC Homography로 오매칭 제거 | [sift_sign_recognition](./sift_sign_recognition) | OpenCV |
| 판정 로직 설계 | 매칭 수와 인라이어 수 기준으로 여러 후보 중 가장 알맞은 것을 선택하고, 기준 미달 시 인식 실패로 처리 | [sift_sign_recognition](./sift_sign_recognition) | OpenCV, NumPy |
| 결과 시각화 | 매칭선, 특징점, 위치 외곽선을 화면 축소 비율에 맞춰 보이게 그려 검증 가능한 결과 화면 구성 | [sift_sign_recognition](./sift_sign_recognition) | OpenCV |
| 영상 처리 프로그램 개발 | 영상 파일을 불러와 프레임 단위로 처리하고, 버튼과 결과 화면이 있는 데스크톱 프로그램으로 구현 | [sift_sign_recognition](./sift_sign_recognition) | PyQt5, OpenCV |
| 영상 처리 이론 | 컨볼루션, 전처리, 모폴로지, 에지 검출, 영역 분할, 미분 필터의 원리 설명 | [theory](./theory) 01, 02 | - |
| 딥러닝 학습 원리 | 손실 함수, 활성 함수, 경사하강법, 역전파, 최적화, 하이퍼파라미터 설명 | [theory](./theory) 03 | - |

## 이론 정리

| 분류 | 주제 | 문서 |
|---|---|---|
| 영상 처리 | 컨볼루션, 이미지 전처리, 모폴로지, 기하학 변환 | [01_convolution_preprocessing_morphology.md](./theory/01_convolution_preprocessing_morphology.md) |
| 에지와 영역 | Edge 검출, 영역 분할, 미분 필터 | [02_edge_segmentation_derivative.md](./theory/02_edge_segmentation_derivative.md) |
| 딥러닝 | 딥러닝 기초 (퍼셉트론, 손실 함수, 역전파, 최적화) | [03_deep_learning_fundamentals.md](./theory/03_deep_learning_fundamentals.md) |

## 실습

| 분류 | 주제 | 폴더 |
|---|---|---|
| 지역 특징 | SIFT 매칭 기반 어린이 보호구역 표지판 인식 | [sift_sign_recognition](./sift_sign_recognition) |

## 폴더 구성



    computer-vision-basics/
    ├─ README.md
    ├─ theory/
    │   ├─ 01_convolution_preprocessing_morphology.md
    │   ├─ 02_edge_segmentation_derivative.md
    │   └─ 03_deep_learning_fundamentals.md
    └─ sift_sign_recognition/