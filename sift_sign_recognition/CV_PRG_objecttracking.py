import sys
import cv2 as cv
import numpy as np
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 코드가 있는 폴더 기준 경로


def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv.imdecode(data, cv.IMREAD_COLOR)
    except Exception:
        return None


def draw_matches_thick(sign_img, sign_kp, road_img, road_kp, matches, box=None,
                       view_w=900, view_h=200, line_px=3):
    # 표지판(왼쪽)과 도로 프레임(오른쪽)을 가로로 이어 붙인 캔버스 생성
    h1, w1 = sign_img.shape[:2]
    h2, w2 = road_img.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = sign_img
    canvas[:h2, w1:] = road_img

    # QLabel에 축소되어 표시되는 비율을 계산해, 축소 후에도 line_px 두께로 보이게 보정
    scale = min(view_w / canvas.shape[1], view_h / canvas.shape[0])
    thick = max(1, int(round(line_px / scale)))

    # 표지판 위치 외곽선(초록색), 도로 프레임은 표지판 너비만큼 오른쪽으로 이동
    if box is not None:
        pts = np.int32(box) + np.array([w1, 0], dtype=np.int32)
        cv.polylines(canvas, [pts], True, (0, 255, 0), thick)

    for m in matches:
        p1 = tuple(map(int, sign_kp[m.queryIdx].pt))
        x2, y2 = map(int, road_kp[m.trainIdx].pt)
        p2 = (x2 + w1, y2)
        cv.line(canvas, p1, p2, (0, 255, 255), thick, cv.LINE_AA)  # 매칭선(노랑)
        cv.circle(canvas, p1, thick * 2, (0, 0, 255), -1)  # 표지판 쪽 특징점(빨강)
        cv.circle(canvas, p2, thick * 2, (0, 0, 255), -1)  # 도로 쪽 특징점(빨강)
    return canvas


class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('어린이 보호구역 인식 시스템')
        self.setGeometry(100, 100, 1300, 750)

        self.signFiles = [
            [os.path.join(BASE_DIR, 'schoolzone.png'), '어린이 보호구역'],
            [os.path.join(BASE_DIR, '30limit.png'), '속도 30 제한'],
            [os.path.join(BASE_DIR, 'crosswalk.png'), '횡단보도']
        ]
        self.signImgs = []
        self.signKps = []
        self.signDes = []

        self.roadImg = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateVideoFrame)

        self.autoRecognize = False
        self.recognition_interval = 10
        self.frame_count = 0
        self.last_result_text = ""

        btn_layout = QHBoxLayout()
        self.signButton = QPushButton('표지판 등록', self)
        self.roadButton = QPushButton('도로 영상 불러오기', self)
        self.recognitionButton = QPushButton('인식 시작', self)
        self.quitButton = QPushButton('나가기', self)

        btn_layout.addWidget(self.signButton)
        btn_layout.addWidget(self.roadButton)
        btn_layout.addWidget(self.recognitionButton)
        btn_layout.addWidget(self.quitButton)

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.signLabels = [QLabel('표지판 1'), QLabel('표지판 2'), QLabel('표지판 3')]
        for lbl in self.signLabels:
            lbl.setFixedSize(300, 200)
            lbl.setStyleSheet('border: 1px solid black; background: #eee;')
            lbl.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(lbl)

        right_layout = QVBoxLayout()
        self.videoLabel = QLabel('도로 영상 영역')
        self.videoLabel.setFixedSize(900, 400)
        self.videoLabel.setStyleSheet('border: 1px solid black; background: #000;')
        self.videoLabel.setAlignment(Qt.AlignCenter)

        self.matchLabel = QLabel('매칭 결과 (특징점)')
        self.matchLabel.setFixedSize(900, 200)
        self.matchLabel.setStyleSheet('border: 1px solid black; background: #fff;')
        self.matchLabel.setAlignment(Qt.AlignCenter)

        self.statusLabel = QLabel('환영합니다! 먼저 "표지판 등록"을 누르세요.')
        right_layout.addWidget(self.videoLabel)
        right_layout.addWidget(self.matchLabel)
        right_layout.addWidget(self.statusLabel)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(btn_layout)
        layout.addLayout(main_layout)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.signButton.clicked.connect(self.signFunction)
        self.roadButton.clicked.connect(self.roadFunction)
        self.recognitionButton.clicked.connect(self.startAutoRecognition)
        self.quitButton.clicked.connect(self.quitFunction)

    def cv_to_pixmap(self, img, w, h):
        if img is None:
            return QPixmap()
        if len(img.shape) == 2:
            h_i, w_i = img.shape
            qimg = QImage(img.data, w_i, h_i, w_i, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h_i, w_i, ch = rgb.shape
        qimg = QImage(rgb.data, w_i, h_i, ch * w_i, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def signFunction(self):
        self.signImgs = []
        self.signKps = []
        self.signDes = []
        sift = cv.SIFT_create()

        ok_count = 0
        for i, (fname, _) in enumerate(self.signFiles):
            img = imread_unicode(fname)
            if img is None:
                self.signLabels[i].setText(f'파일 없음\n{fname}')
                continue

            self.signImgs.append(img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            self.signKps.append(kp)
            self.signDes.append(des)
            self.signLabels[i].setPixmap(self.cv_to_pixmap(img, 300, 200))
            ok_count += 1

        self.statusLabel.setText(f'표지판 등록 완료: {ok_count}/3')
        self.matchLabel.clear()
        self.matchLabel.setText('매칭 결과 (특징점)')

    def roadFunction(self):
        fname = QFileDialog.getOpenFileName(
            self, '도로 영상 선택', './',
            'Video Files (*.mp4 *.avi *.mov *.MOV);;All Files (*)'
        )[0]
        if not fname:
            return

        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None

        self.cap = cv.VideoCapture(fname)
        if not self.cap.isOpened():
            self.statusLabel.setText('도로 영상을 열 수 없습니다.')
            return

        ret, frame = self.cap.read()
        if not ret:
            self.statusLabel.setText('도로 영상 첫 프레임을 읽지 못했습니다.')
            return

        self.roadImg = frame.copy()
        self.videoLabel.setPixmap(self.cv_to_pixmap(frame, 900, 400))
        self.statusLabel.setText('도로 영상 불러오기 완료. "인식 시작"을 누르면 자동 인식됩니다.')
        self.autoRecognize = False
        self.frame_count = 0

    def startAutoRecognition(self):
        if len(self.signImgs) == 0:
            self.statusLabel.setText('먼저 표지판을 등록하세요.')
            return
        if self.cap is None:
            self.statusLabel.setText('먼저 도로 영상을 불러오세요.')
            return

        self.autoRecognize = True
        self.statusLabel.setText('자동 인식이 시작되었습니다.')
        if not self.timer.isActive():
            self.timer.start(30)

    def updateVideoFrame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.statusLabel.setText('도로 영상 재생 종료 또는 읽기 실패.')
                return

        self.roadImg = frame.copy()
        self.videoLabel.setPixmap(self.cv_to_pixmap(frame, 900, 400))

        if self.autoRecognize:
            self.frame_count += 1
            if self.frame_count % self.recognition_interval == 0:
                self.recognitionOnce()

    def recognitionOnce(self):
        if len(self.signImgs) == 0 or self.roadImg is None:
            return

        sift = cv.SIFT_create()
        road_gray = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        road_kp, road_des = sift.detectAndCompute(road_gray, None)

        if road_des is None or len(road_kp) == 0:
            return

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        best_idx = -1
        best_good = []
        best_sign_kp = None
        best_H = None
        best_inliers = 0

        RATIO_T = 0.55
        MIN_GOOD = 8
        MIN_INLIERS = 6

        for i in range(len(self.signImgs)):
            if self.signDes[i] is None or len(self.signKps[i]) == 0:
                continue

            try:
                knn_match = matcher.knnMatch(self.signDes[i], road_des, 2)
            except cv.error:
                continue

            good = []
            for pair in knn_match:
                if len(pair) < 2:
                    continue
                m, n = pair
                if n.distance == 0:
                    continue
                if m.distance < RATIO_T * n.distance:
                    good.append(m)

            if len(good) < MIN_GOOD:
                continue

            src_pts = np.float32([self.signKps[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([road_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_idx = i
                best_good = good
                best_sign_kp = self.signKps[i]
                best_H = H

        if best_idx == -1 or best_inliers < MIN_INLIERS:
            self.matchLabel.setText('주어진 표지판이 아닙니다.\n매칭 조건을 통과하지 못했습니다.')
            self.statusLabel.setText('자동 인식 실패: 주어진 표지판이 아닙니다.')
            return

        h1, w1 = self.signImgs[best_idx].shape[:2]
        box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
        box2 = cv.perspectiveTransform(box1, best_H)

        img_match = draw_matches_thick(
            self.signImgs[best_idx], best_sign_kp,
            self.roadImg, road_kp,
            best_good, box2
        )

        self.matchLabel.setPixmap(self.cv_to_pixmap(img_match, 900, 200))
        self.statusLabel.setText(f'자동 인식 성공: {self.signFiles[best_idx][1]}')

    def quitFunction(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrafficWeak()
    win.show()
    sys.exit(app.exec_())