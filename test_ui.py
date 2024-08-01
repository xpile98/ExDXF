import sys
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QListWidget, \
    QMessageBox, QMenuBar, QAction, QFileDialog, QListWidgetItem, QAbstractItemView, QSlider, QLabel, QDialog
from PyQt5.QtCore import Qt, QTimer
import numpy as np


class DXFViewer(QMainWindow):
    def __init__(self, dxf_path=None):
        super().__init__()
        self.setWindowTitle('DXF Viewer')
        self.setGeometry(100, 100, 1200, 800)

        self.dxf_path = dxf_path
        self.doc = None
        self.msp = None
        self.entities = []
        self.entity_visibility = []
        self.show_direction = True
        self.show_order = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.simulation_step = 0
        self.simulating = False
        self.path_points = []  # 시뮬레이션 경로를 저장할 리스트
        self.current_point_index = 0  # 현재 이동 중인 점의 인덱스
        self.current_path = []  # 현재 선의 모든 점들
        self.speed = 8.0  # 이동 속도 (단위 시간당 거리)

        self.initUI()

        if dxf_path:
            self.load_dxf(dxf_path)

    def initUI(self):
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)

        file_menu = self.menuBar.addMenu('File')

        load_action = QAction('Load DXF', self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        save_action = QAction('Save As', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        view_menu = self.menuBar.addMenu('View')

        self.direction_action = QAction('Show Direction', self, checkable=True)
        self.direction_action.setChecked(True)
        self.direction_action.triggered.connect(self.toggle_show_direction)
        view_menu.addAction(self.direction_action)

        self.order_action = QAction('Show Order', self, checkable=True)
        self.order_action.setChecked(True)
        self.order_action.triggered.connect(self.toggle_show_order)
        view_menu.addAction(self.order_action)

        # Simulate action 추가
        self.simulate_action = QAction('Simulate', self)
        self.simulate_action.triggered.connect(self.start_simulation)
        view_menu.addAction(self.simulate_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_widget.itemChanged.connect(self.update_entity_visibility)
        left_layout.addWidget(self.list_widget)

        button_layout = QVBoxLayout()

        up_button = QPushButton('Move Up')
        up_button.clicked.connect(self.move_up)
        button_layout.addWidget(up_button)

        down_button = QPushButton('Move Down')
        down_button.clicked.connect(self.move_down)
        button_layout.addWidget(down_button)

        delete_button = QPushButton('Delete Selected')
        delete_button.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_button)

        reverse_button = QPushButton('Reverse Order Selected')
        reverse_button.clicked.connect(self.reverse_order_selected)
        button_layout.addWidget(reverse_button)

        apply_button = QPushButton('Apply Order')
        apply_button.clicked.connect(self.apply_order)
        button_layout.addWidget(apply_button)

        left_layout.addLayout(button_layout)

        self.canvas = FigureCanvas(plt.Figure())
        right_layout.addWidget(self.canvas)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 4)

    def load_dxf(self, file_path):
        self.doc = ezdxf.readfile(file_path)
        self.msp = self.doc.modelspace()
        self.entities = list(self.msp.query('LWPOLYLINE LINE'))
        self.entity_visibility = [True] * len(self.entities)
        self.update_list_widget()
        self.visualize_entities()

    def update_list_widget(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, entity in enumerate(self.entities):
            layer_name = entity.dxf.layer
            entity_type = entity.dxftype()
            item = QListWidgetItem(f'{entity_type} {i + 1} (Layer: {layer_name})')
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if self.entity_visibility[i] else Qt.Unchecked)
            item.setData(Qt.UserRole, i)  # Custom data to store entity index
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def update_entity_visibility(self, item):
        index = item.data(Qt.UserRole)
        self.entity_visibility[index] = item.checkState() == Qt.Checked
        self.visualize_entities()

    def move_up(self):
        current_row = self.list_widget.currentRow()
        if current_row > 0:
            current_item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row - 1, current_item)
            self.list_widget.setCurrentRow(current_row - 1)
            self.swap_entities(current_row, current_row - 1)
            self.update_list_widget_selection(current_row - 1)

    def move_down(self):
        current_row = self.list_widget.currentRow()
        if current_row < self.list_widget.count() - 1:
            current_item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row + 1, current_item)
            self.list_widget.setCurrentRow(current_row + 1)
            self.swap_entities(current_row, current_row + 1)
            self.update_list_widget_selection(current_row + 1)

    def swap_entities(self, index1, index2):
        self.entities[index1], self.entities[index2] = self.entities[index2], self.entities[index1]
        self.entity_visibility[index1], self.entity_visibility[index2] = self.entity_visibility[index2], self.entity_visibility[index1]

    def update_list_widget_selection(self, new_row):
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(new_row)
        self.list_widget.blockSignals(False)
        self.visualize_entities()

    def delete_selected(self):
        selected_items = self.list_widget.selectedItems()
        for item in selected_items:
            row = self.list_widget.row(item)
            del self.entities[row]
            del self.entity_visibility[row]
            self.list_widget.takeItem(row)
        self.visualize_entities()

    def reverse_order_selected(self):
        current_row = self.list_widget.currentRow()
        if current_row != -1:
            entity = self.entities[current_row]
            if entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points('xy')
                reversed_points = list(reversed(points))
                entity.set_points(reversed_points)
            self.visualize_entities()

    def apply_order(self):
        order = [self.list_widget.item(i).data(Qt.UserRole) for i in range(self.list_widget.count())]
        self.entities = [self.entities[i] for i in order]
        self.entity_visibility = [self.entity_visibility[i] for i in order]
        self.update_list_widget()
        self.visualize_entities()

    def toggle_show_direction(self):
        self.show_direction = self.direction_action.isChecked()
        self.visualize_entities()

    def toggle_show_order(self):
        self.show_order = self.order_action.isChecked()
        self.visualize_entities()

    def visualize_entities(self):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        for idx, (entity, visible) in enumerate(zip(self.entities, self.entity_visibility)):
            if not visible:
                continue
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                x, y = zip(*vertices)
                ax.plot(x, y, color='black')
                if self.show_direction:
                    for i in range(len(vertices) - 1):
                        start = vertices[i]
                        end = vertices[i + 1]
                        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                                 head_width=2, head_length=2, fc='red', ec='red')
                if self.show_order:
                    start_x, start_y = vertices[0]
                    ax.text(start_x, start_y, str(idx + 1), fontsize=12, color='blue', ha='right')
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                ax.plot([start.x, end.x], [start.y, end.y], color='black')
                if self.show_direction:
                    ax.arrow(start.x, start.y, end.x - start.x, end.y - start.y,
                             head_width=2, head_length=2, fc='red', ec='red')
                if self.show_order:
                    ax.text(start.x, start.y, str(idx + 1), fontsize=12, color='blue', ha='right')

        ax.set_title('DXF Entities with Order')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        self.canvas.draw()

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DXF File", "", "DXF Files (*.dxf);;All Files (*)",
                                                   options=options)
        if file_path:
            self.load_dxf(file_path)

    def save_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save DXF File", "", "DXF Files (*.dxf);;All Files (*)",
                                                   options=options)
        if file_path:
            new_doc = ezdxf.new(dxfversion='R2010')
            new_msp = new_doc.modelspace()
            for entity in self.entities:
                if entity.dxftype() == 'LWPOLYLINE':
                    new_msp.add_lwpolyline(entity.get_points('xy'), dxfattribs=entity.dxfattribs())
                elif entity.dxftype() == 'LINE':
                    new_msp.add_line(entity.dxf.start, entity.dxf.end, dxfattribs=entity.dxfattribs())
            new_doc.saveas(file_path)
            QMessageBox.information(self, "Saved", f"File saved to {file_path}")

    def change_speed(self, value):
        self.speed = value / 10.0  # 슬라이더 값을 0.1 단위로 변환
        self.update_simulation_path()  # 속도 변경 시 경로 재보간

    def update_simulation_path(self):
        if self.simulating:
            # 현재 위치를 기준으로 남은 경로를 재보간
            if self.simulation_step < len(self.current_path):
                current_position = self.current_path[self.simulation_step]
            else:
                current_position = self.current_path[-1]

            # 현재 위치에서 남은 경로 계산
            remaining_path = [current_position] + self.path_points[self.current_point_index + 1:]

            # 현재 위치에서 경로 재보간
            self.current_path = self.interpolate_points(remaining_path, self.speed)

            # 현재 위치 업데이트
            self.simulation_step = 0

    def start_simulation(self):
        if not self.simulating:
            self.simulating = True
            self.simulation_step = 0
            self.path_points = self.get_path_points()  # 시뮬레이션 경로 포인트를 가져옴
            self.current_point_index = 0
            self.current_path = self.interpolate_points(self.path_points, self.speed)  # 초기 경로 보간
            self.show_slider_dialog()  # 슬라이더 팝업 표시
            self.timer.start(10)  # 100ms 간격으로 타이머 실행

    def get_path_points(self):
        path_points = []
        for idx, entity in enumerate(self.entities):
            if not self.entity_visibility[idx]:
                continue
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                path_points.extend(vertices)
            elif entity.dxftype() == 'LINE':
                path_points.append((entity.dxf.start.x, entity.dxf.start.y))
                path_points.append((entity.dxf.end.x, entity.dxf.end.y))
        return path_points

    def interpolate_points(self, points, speed):
        """주어진 점들 사이를 일정 속도로 이동할 수 있도록 보간된 점들을 생성"""
        interpolated_points = []
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            distance = np.linalg.norm(end - start)
            if distance > 0:
                num_steps = max(int(distance / speed), 1)  # 최소 한 스텝은 필요
                for step in range(num_steps):
                    interpolated_point = start + (end - start) * (step / num_steps)
                    interpolated_points.append(interpolated_point)
            # 마지막 점만 추가하여 다음 선의 시작점으로 바로 점프
            interpolated_points.append(end)
        return interpolated_points

    def update_simulation(self):
        # 시뮬레이션이 끝났는지 확인
        if self.simulation_step >= len(self.current_path):
            self.timer.stop()
            self.simulating = False
            self.slider_dialog.close()  # 슬라이더 팝업 닫기
            QMessageBox.information(self, "Simulation", "Simulation finished!")
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # 모든 선을 검정색으로 그리기
        for idx, entity in enumerate(self.entities):
            if not self.entity_visibility[idx]:
                continue
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                x, y = zip(*vertices)
                ax.plot(x, y, color='black')
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                ax.plot([start.x, end.x], [start.y, end.y], color='black')

        # 레이저 효과를 주기 위해 현재 경로 포인트에 레이저 스타일로 그리기
        if self.current_path:
            x, y = self.current_path[self.simulation_step]

            # 레이저 빛나는 효과를 점선으로 표현
            ax.plot(x, y, 'ro', markersize=5, markeredgewidth=2, markeredgecolor='darkred', alpha=0.8)
            ax.plot(x, y, 'ro', markersize=10, markeredgewidth=1, markeredgecolor='red', alpha=0.5)

            # 레이저 꼬리 효과
            if self.simulation_step > 0:
                previous_points = self.current_path[max(0, self.simulation_step - 5):self.simulation_step]
                prev_x, prev_y = zip(*previous_points)
                ax.plot(prev_x, prev_y, 'r--', linewidth=5, alpha=0.5)  # 점선으로 레이저 경로 표현

            self.current_position = (x, y)  # 현재 위치 업데이트

        ax.set_title('DXF Entities Simulation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        self.canvas.draw()

        # 다음 선으로 바로 점프할 때 simulation_step을 즉시 증가
        if self.simulation_step < len(self.current_path) - 1:
            next_point = self.current_path[self.simulation_step + 1]
            if np.allclose(np.array(next_point), np.array((x, y)), atol=1e-7):
                # 다음 점이 같은 위치라면 한 번만 증가
                self.simulation_step += 1
            else:
                # 다른 위치라면 다음 선의 시작점으로 즉시 점프
                self.simulation_step += 1
                while self.simulation_step < len(self.current_path) - 1:
                    current_point = self.current_path[self.simulation_step]
                    next_point = self.current_path[self.simulation_step + 1]
                    if not np.allclose(np.array(current_point), np.array(next_point), atol=1e-7):
                        break
                    self.simulation_step += 1
        else:
            # 마지막 점에 도달했을 때 증가
            self.simulation_step += 1

    def show_slider_dialog(self):
        self.slider_dialog = QDialog(self)
        self.slider_dialog.setWindowTitle('Adjust Speed')
        layout = QVBoxLayout()

        slider_label = QLabel('Speed')
        layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(80)  # 초기 속도 설정
        self.slider.valueChanged.connect(self.change_speed)
        layout.addWidget(self.slider)

        self.slider_dialog.setLayout(layout)
        self.slider_dialog.setWindowModality(Qt.ApplicationModal)
        self.slider_dialog.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dxf_path = 'test3.dxf'  # 기본 DXF 파일 경로
    viewer = DXFViewer(dxf_path)
    viewer.show()
    sys.exit(app.exec_())
