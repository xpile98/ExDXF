import sys
import ezdxf
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QListWidget, \
    QMessageBox, QMenuBar, QAction, QFileDialog, QListWidgetItem, QAbstractItemView, QSlider, QLabel, QDialog, QInputDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
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
        self.selected_entity_index = None  # 선택된 엔티티 인덱스를 저장

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.simulation_step = 0
        self.simulating = False
        self.path_points = []  # 시뮬레이션 경로를 저장할 리스트
        self.current_point_index = 0  # 현재 이동 중인 점의 인덱스
        self.current_path = []  # 현재 선의 모든 점들
        self.speed = 8.0  # 이동 속도 (단위 시간당 거리)

        # 기본 색상 설정
        self.line_color = 'white'
        self.selected_line_color = (144, 238, 144)
        self.arrow_color = 'red'
        self.order_text_color = (144, 238, 144)

        self.initUI()

        if dxf_path:
            self.load_dxf(dxf_path)

    def initUI(self):
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)

        file_menu = self.menuBar.addMenu('파일')

        load_action = QAction('DXF 파일 불러오기', self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        save_action = QAction('다른 이름으로 저장', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        view_menu = self.menuBar.addMenu('보기')

        self.direction_action = QAction('선 방향 표시', self, checkable=True)
        self.direction_action.setChecked(True)
        self.direction_action.triggered.connect(self.toggle_show_direction)
        view_menu.addAction(self.direction_action)

        self.order_action = QAction('선 순서 표시', self, checkable=True)
        self.order_action.setChecked(True)
        self.order_action.triggered.connect(self.toggle_show_order)
        view_menu.addAction(self.order_action)

        # Simulate action 추가
        self.simulate_action = QAction('레이저 커팅 시뮬레이션', self)
        self.simulate_action.triggered.connect(self.start_simulation)
        view_menu.addAction(self.simulate_action)

        # Add Show All Lines and Hide All Lines to the View menu
        show_all_action = QAction('모든 선 표시', self)
        show_all_action.triggered.connect(self.show_all_entities)
        view_menu.addAction(show_all_action)

        hide_all_action = QAction('모든 선 숨기기', self)
        hide_all_action.triggered.connect(self.hide_all_entities)
        view_menu.addAction(hide_all_action)

        edit_menu = self.menuBar.addMenu('편집')

        # initUI 내에 메뉴 액션 추가
        job_editor_action = QAction('커팅 도면 편집', self)
        job_editor_action.triggered.connect(self.open_job_editor)
        edit_menu.addAction(job_editor_action)

        top_left_sort_action = QAction('정렬: 좌상단', self)
        top_left_sort_action.triggered.connect(self.sort_top_left)
        edit_menu.addAction(top_left_sort_action)

        bottom_right_sort_action = QAction('정렬: 우하단', self)
        bottom_right_sort_action.triggered.connect(self.sort_bottom_right)
        edit_menu.addAction(bottom_right_sort_action)


        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)  # 단일 선택 모드로 변경
        self.list_widget.itemChanged.connect(self.update_entity_visibility)
        left_layout.addWidget(self.list_widget)

        button_layout = QVBoxLayout()

        change_order_button = QPushButton('선 순서 변경')
        change_order_button.clicked.connect(self.change_order)
        button_layout.addWidget(change_order_button)

        reverse_button = QPushButton('선 방향 반전')
        reverse_button.clicked.connect(self.reverse_order_selected)
        button_layout.addWidget(reverse_button)

        delete_button = QPushButton('선 삭제')
        delete_button.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_button)

        init_button = QPushButton('초기화')
        init_button.clicked.connect(self.init_order)
        button_layout.addWidget(init_button)

        left_layout.addLayout(button_layout)

        # PyQtGraph PlotWidget으로 변경
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True)

        # Ensure that the aspect ratio does not lock the scaling of axes
        self.plot_widget.setAspectLocked(True)

        self.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)  # 마우스 클릭 이벤트 연결
        right_layout.addWidget(self.plot_widget)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 4)

    def load_dxf(self, file_path):
        self.doc = ezdxf.readfile(file_path)
        self.msp = self.doc.modelspace()
        self.entities = list(self.msp.query('LWPOLYLINE LINE'))
        self.entity_visibility = [True] * len(self.entities)
        self.dxf_path = file_path  # Store the current file path
        self.setWindowTitle(f'LSTech DXF Editor  [{file_path}]')  # Update window title with file name
        self.update_list_widget()
        self.visualize_entities()

    def show_all_entities(self):
        self.list_widget.blockSignals(True)  # 시그널 차단하여 불필요한 이벤트 방지
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)
            self.entity_visibility[i] = True
        self.list_widget.blockSignals(False)
        self.visualize_entities()

    def hide_all_entities(self):
        self.list_widget.blockSignals(True)  # 시그널 차단하여 불필요한 이벤트 방지
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
            self.entity_visibility[i] = False
        self.list_widget.blockSignals(False)
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

    def change_order(self):
        # 현재 선택된 항목의 인덱스를 가져옴
        current_row = self.list_widget.currentRow()
        if current_row == -1:
            QMessageBox.warning(self, "Warning", "Please select an entity to swap.")
            return

        # 다른 엔티티를 선택하기 위한 대화상자 표시
        item_names = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
        target_item, ok = QInputDialog.getItem(self, "Select Entity", "Select entity to swap with:", item_names, 0,
                                               False)

        if ok and target_item:
            target_row = item_names.index(target_item)

            # Swap entities in the data model
            self.swap_entities(current_row, target_row)

            # Swap entities in the list widget
            current_item = self.list_widget.takeItem(current_row)
            target_item = self.list_widget.takeItem(target_row - 1 if current_row < target_row else target_row)
            self.list_widget.insertItem(current_row, target_item)
            self.list_widget.insertItem(target_row, current_item)
            self.list_widget.setCurrentRow(target_row)

            self.update_list_widget_selection(target_row)

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

    def init_order(self):
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
        self.plot_widget.clear()

        tolerance = 1  # Customize this value as needed

        for idx, (entity, visible) in enumerate(zip(self.entities, self.entity_visibility)):
            if not visible:
                continue
            color = self.selected_line_color if idx == self.selected_entity_index else self.line_color
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                x, y = zip(*vertices)
                self.plot_widget.plot(x, y, pen=color)
                if self.show_direction and len(vertices) > 1:
                    # 첫 번째 선분의 방향을 첫 번째 점에서 두 번째 점으로
                    start = vertices[0]
                    end = vertices[1]
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]

                    # 기울기를 기준으로 수평과 수직을 구분
                    if np.isclose(dy, 0, atol=tolerance):  # 수평선
                        if dx > 0:
                            arrow_angle = 180  # 오른쪽
                        else:
                            arrow_angle = 0  # 왼쪽
                    elif np.isclose(dx, 0, atol=tolerance):  # 수직선
                        if dy > 0:
                            arrow_angle = 90  # 위쪽
                        else:
                            arrow_angle = -90  # 아래쪽
                    else:
                        # 대각선 방향을 처리 (45도 간격으로)
                        angle = np.degrees(np.arctan2(dy, dx))
                        arrow_angle = angle

                    arrow = pg.ArrowItem(pos=(start[0], start[1]), angle=arrow_angle, tipAngle=45, baseAngle=30,
                                         headLen=15,
                                         tailLen=0, brush=self.arrow_color)
                    self.plot_item.addItem(arrow)
                if self.show_order:
                    start_x, start_y = vertices[0]
                    text = pg.TextItem(text=str(idx + 1), color=self.order_text_color, anchor=(0, 0))
                    text.setPos(start_x, start_y)

                    # Create a font object and set the size
                    font = QFont()
                    font.setPointSize(18)  # 글자 크기를 18pt로 설정
                    text.setFont(font)

                    self.plot_item.addItem(text)
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                self.plot_widget.plot([start.x, end.x], [start.y, end.y], pen=color)
                if self.show_direction:
                    # 선분의 방향을 첫 번째 점에서 두 번째 점으로
                    dx = end.x - start.x
                    dy = end.y - start.y

                    # 기울기를 기준으로 수평과 수직을 구분
                    if np.isclose(dy, 0, atol=tolerance):  # 수평선
                        if dx > 0:
                            arrow_angle = 180  # 오른쪽
                        else:
                            arrow_angle = 0  # 왼쪽
                    elif np.isclose(dx, 0, atol=tolerance):  # 수직선
                        if dy > 0:
                            arrow_angle = 90  # 위쪽
                        else:
                            arrow_angle = -90  # 아래쪽
                    else:
                        # 대각선 방향을 처리 (45도 간격으로)
                        angle = np.degrees(np.arctan2(dy, dx))
                        arrow_angle = angle

                    arrow = pg.ArrowItem(pos=(start.x, start.y), angle=arrow_angle, tipAngle=45, baseAngle=30,
                                         headLen=15,
                                         tailLen=0, brush=self.arrow_color)
                    self.plot_item.addItem(arrow)
                if self.show_order:
                    text = pg.TextItem(text=str(idx + 1), color=self.order_text_color, anchor=(0, 0))
                    text.setPos(start.x, start.y)
                    self.plot_item.addItem(text)

    def on_canvas_click(self, event):
        pos = event.scenePos()
        click_x, click_y = self.plot_item.vb.mapSceneToView(pos).x(), self.plot_item.vb.mapSceneToView(pos).y()

        closest_entity_idx = None
        min_distance = float('inf')

        for idx, (entity, visible) in enumerate(zip(self.entities, self.entity_visibility)):
            if not visible:
                continue
            distance = self.calculate_distance_to_entity(click_x, click_y, entity)
            if distance < min_distance:
                min_distance = distance
                closest_entity_idx = idx

        if closest_entity_idx is not None:
            self.select_entity_in_list(closest_entity_idx)

    def calculate_distance_to_entity(self, click_x, click_y, entity):
        # 클릭된 위치와 엔티티 사이의 최단 거리 계산
        if entity.dxftype() == 'LWPOLYLINE':
            vertices = entity.get_points('xy')
        elif entity.dxftype() == 'LINE':
            vertices = [(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]
        else:
            return float('inf')  # 지원하지 않는 엔티티 타입에 대해서는 무한대 거리

        return self.point_to_polyline_distance((click_x, click_y), vertices)

    def point_to_polyline_distance(self, point, vertices):
        """점과 폴리라인(다각형) 사이의 최단 거리 계산"""
        px, py = point
        min_distance = float('inf')

        for i in range(len(vertices) - 1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i + 1]
            distance = self.point_to_segment_distance(px, py, x1, y1, x2, y2)
            min_distance = min(min_distance, distance)

        return min_distance

    def point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """점과 선분 사이의 최단 거리 계산"""
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)  # 선분의 시작점과 끝점이 같은 경우

        # 점 투영 벡터 계산
        projection = np.dot(point_vec, line_vec) / line_len
        if projection < 0:
            closest_point = np.array([x1, y1])
        elif projection > 1:
            closest_point = np.array([x2, y2])
        else:
            closest_point = np.array([x1, y1]) + projection * line_vec

        # 점과 가장 가까운 점 사이의 거리 계산
        return np.linalg.norm(np.array([px, py]) - closest_point)

    def select_entity_in_list(self, index):
        self.selected_entity_index = index
        self.update_list_widget_selection(index)

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

            # "View All" 기능을 통해 모든 엔티티가 보이도록 설정
            self.plot_widget.getViewBox().autoRange()  # View all entities

            # 방향 및 순서 표시 숨기기
            self.direction_action.setChecked(False)
            self.order_action.setChecked(False)

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

            # 시뮬레이션 종료 후 원래 보기 옵션 복원
            self.direction_action.setChecked(True)
            self.order_action.setChecked(True)
            self.visualize_entities()
            return

        self.plot_widget.clear()

        # 모든 선을 검정색으로 그리기
        for idx, entity in enumerate(self.entities):
            if not self.entity_visibility[idx]:
                continue
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                x, y = zip(*vertices)
                self.plot_widget.plot(x, y, pen = self.line_color)
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                self.plot_widget.plot([start.x, end.x], [start.y, end.y], pen = self.line_color)

        # 레이저 효과를 주기 위해 현재 경로 포인트에 레이저 스타일로 그리기
        if self.current_path:
            x, y = self.current_path[self.simulation_step]

            # 레이저 빛나는 효과를 점선으로 표현
            laser_point = pg.ScatterPlotItem([x], [y], pen=pg.mkPen('darkred'), brush=pg.mkBrush('red'), size=10)
            self.plot_item.addItem(laser_point)

            # 레이저 꼬리 효과
            if self.simulation_step > 0:
                previous_points = self.current_path[max(0, self.simulation_step - 5):self.simulation_step]
                if previous_points:  # Ensure there are previous points
                    tail_x, tail_y = zip(*previous_points)
                    tail_x = np.array(tail_x)  # Convert to 1D ndarray
                    tail_y = np.array(tail_y)  # Convert to 1D ndarray
                    tail = pg.PlotCurveItem(tail_x, tail_y, pen=pg.mkPen('red', style=Qt.DashLine))
                    self.plot_item.addItem(tail)

            self.current_position = (x, y)  # 현재 위치 업데이트

        self.plot_widget.setTitle('DXF Entities Simulation')
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')

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
        self.slider.setMinimum(50)
        self.slider.setMaximum(150)
        self.slider.setValue(80)  # 초기 속도 설정
        self.slider.valueChanged.connect(self.change_speed)
        layout.addWidget(self.slider)

        self.slider_dialog.setLayout(layout)
        self.slider_dialog.setWindowModality(Qt.ApplicationModal)
        self.slider_dialog.show()

    # 메인 윈도우에 JobEditorDialog를 추가하는 부분
    def open_job_editor(self):
        dialog = JobEditorDialog(self.entities, self)
        dialog.exec_()

    def sort_top_left(self):
        horizontal_lines = [e for e in self.entities if self.is_horizontal(e)]
        vertical_lines = [e for e in self.entities if self.is_vertical(e)]

        # Sort horizontal lines by y first (bottom to top), then by x (left to right)
        horizontal_lines.sort(key=lambda e: (
            -e.dxf.start.y if e.dxftype() == 'LINE' else -e.get_points('xy')[0][1],
            e.dxf.start.x if e.dxftype() == 'LINE' else e.get_points('xy')[0][0]
        ))

        # Sort vertical lines by x first (left to right), then by y (top to bottom)
        vertical_lines.sort(key=lambda e: (
            e.dxf.start.x if e.dxftype() == 'LINE' else e.get_points('xy')[0][0],
            e.dxf.start.y if e.dxftype() == 'LINE' else e.get_points('xy')[0][1]
        ))

        # Reverse direction of horizontal lines if needed
        for entity in horizontal_lines:
            self.ensure_direction(entity, left_to_right=True)

        # Reverse direction of vertical lines if needed
        for entity in vertical_lines:
            self.ensure_direction(entity, top_to_bottom=True)

        # Combine sorted lines
        self.entities = horizontal_lines + vertical_lines
        self.update_list_widget()
        self.visualize_entities()

    def sort_bottom_right(self):
        horizontal_lines = [e for e in self.entities if self.is_horizontal(e)]
        vertical_lines = [e for e in self.entities if self.is_vertical(e)]

        # Sort horizontal lines by y first (top to bottom), then by x (right to left)
        horizontal_lines.sort(key=lambda e: (
            e.dxf.start.y if e.dxftype() == 'LINE' else e.get_points('xy')[0][1],
            -e.dxf.start.x if e.dxftype() == 'LINE' else -e.get_points('xy')[0][0]
        ))

        # Sort vertical lines by x first (right to left), then by y (bottom to top)
        vertical_lines.sort(key=lambda e: (
            -e.dxf.start.x if e.dxftype() == 'LINE' else -e.get_points('xy')[0][0],
            -e.dxf.start.y if e.dxftype() == 'LINE' else -e.get_points('xy')[0][1]
        ))

        # Reverse direction of horizontal lines if needed
        for entity in horizontal_lines:
            self.ensure_direction(entity, left_to_right=False)

        # Reverse direction of vertical lines if needed
        for entity in vertical_lines:
            self.ensure_direction(entity, top_to_bottom=False)

        # Combine sorted lines
        self.entities = horizontal_lines + vertical_lines
        self.update_list_widget()
        self.visualize_entities()

    def ensure_direction(self, entity, left_to_right=True, top_to_bottom=True):
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            if self.is_horizontal(entity):
                # Horizontal line
                if left_to_right and start.x > end.x:
                    # Reverse direction if not left-to-right
                    entity.dxf.start, entity.dxf.end = end, start
                elif not left_to_right and start.x < end.x:
                    # Reverse direction if not right-to-left
                    entity.dxf.start, entity.dxf.end = end, start
            elif self.is_vertical(entity):
                # Vertical line
                if top_to_bottom and start.y > end.y:
                    # Reverse direction if not top-to-bottom
                    entity.dxf.start, entity.dxf.end = end, start
                elif not top_to_bottom and start.y < end.y:
                    # Reverse direction if not bottom-to-top
                    entity.dxf.start, entity.dxf.end = end, start

        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xy')
            start = points[0]
            end = points[1]  # Check the direction based on the first segment
            if self.is_horizontal(entity):
                # Horizontal line
                if left_to_right and start[0] > end[0]:
                    entity.set_points(list(reversed(points)))
                elif not left_to_right and start[0] < end[0]:
                    entity.set_points(list(reversed(points)))
            elif self.is_vertical(entity):
                # Vertical line
                if top_to_bottom and start[1] < end[1]:
                    entity.set_points(list(reversed(points)))
                elif not top_to_bottom and start[1] > end[1]:
                    entity.set_points(list(reversed(points)))

    def is_horizontal(self, entity, tolerance=1.0):
        if entity.dxftype() == 'LINE':
            return np.isclose(entity.dxf.start.y, entity.dxf.end.y, atol=tolerance)
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xy')
            return all(np.isclose(y1, y2, atol=tolerance) for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]))
        return False

    def is_vertical(self, entity, tolerance=1.0):
        if entity.dxftype() == 'LINE':
            return np.isclose(entity.dxf.start.x, entity.dxf.end.x, atol=tolerance)
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xy')
            return all(np.isclose(x1, x2, atol=tolerance) for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]))
        return False


class JobEditorDialog(QDialog):
    def __init__(self, entities, parent=None):
        super().__init__(parent)
        self.setWindowTitle('커팅 도면 편집창')
        self.entities = entities
        self.job_list = []
        self.setGeometry(100, 100, 600, 600)

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        # Add labels for the list widgets
        all_entities_layout = QVBoxLayout()
        all_entities_label = QLabel('선 정보')
        all_entities_layout.addWidget(all_entities_label)

        # List widget to display all entities
        self.all_entities_list = QListWidget()
        for i, entity in enumerate(self.entities):
            layer_name = entity.dxf.layer
            entity_type = entity.dxftype()
            item = QListWidgetItem(f'{entity_type} {i + 1} (Layer: {layer_name})')
            item.setData(Qt.UserRole, i)  # Store the index
            self.all_entities_list.addItem(item)
        all_entities_layout.addWidget(self.all_entities_list)

        layout.addLayout(all_entities_layout)

        # Add a label for the job list
        job_list_layout = QVBoxLayout()
        job_list_label = QLabel('커팅 순서')
        job_list_layout.addWidget(job_list_label)

        # List widget to display job list
        self.job_list_widget = QListWidget()
        job_list_layout.addWidget(self.job_list_widget)

        layout.addLayout(job_list_layout)

        # Button layout
        button_layout = QVBoxLayout()

        add_button = QPushButton('커팅 순서에 추가')
        add_button.clicked.connect(self.add_to_job_list)
        button_layout.addWidget(add_button)

        remove_button = QPushButton('커팅 순서에서 삭제')
        remove_button.clicked.connect(self.remove_from_job_list)
        button_layout.addWidget(remove_button)

        add_all_button = QPushButton('모든 선을 커팅 순서에 추가')
        add_all_button.clicked.connect(self.add_all_to_job_list)
        button_layout.addWidget(add_all_button)

        save_button = QPushButton('저장')
        save_button.clicked.connect(self.save_dxf)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def add_to_job_list(self):
        selected_items = self.all_entities_list.selectedItems()
        for item in selected_items:
            entity_index = item.data(Qt.UserRole)
            self.job_list.append(self.entities[entity_index])

            layer_name = self.entities[entity_index].dxf.layer
            entity_type = self.entities[entity_index].dxftype()
            job_item = QListWidgetItem(f'{entity_type} {entity_index + 1} (Layer: {layer_name})')
            job_item.setData(Qt.UserRole, entity_index)
            self.job_list_widget.addItem(job_item)

    def remove_from_job_list(self):
        selected_items = self.job_list_widget.selectedItems()
        for item in selected_items:
            index = self.job_list_widget.row(item)
            del self.job_list[index]
            self.job_list_widget.takeItem(index)

    def add_all_to_job_list(self):
        # Iterate over all entities and add them to the job list
        for i, entity in enumerate(self.entities):
            # Append the entity to the job list
            self.job_list.append(entity)

            # Add a new item to the job list widget with updated information
            layer_name = entity.dxf.layer
            entity_type = entity.dxftype()
            job_item = QListWidgetItem(f'{entity_type} {i + 1} (Layer: {layer_name})')
            job_item.setData(Qt.UserRole, i)
            self.job_list_widget.addItem(job_item)

    def save_dxf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save DXF File", "", "DXF Files (*.dxf);;All Files (*)")
        if file_path:
            new_doc = ezdxf.new(dxfversion='R2010')
            new_msp = new_doc.modelspace()
            for entity in self.job_list:
                if entity.dxftype() == 'LWPOLYLINE':
                    new_msp.add_lwpolyline(entity.get_points('xy'), dxfattribs=entity.dxfattribs())
                elif entity.dxftype() == 'LINE':
                    new_msp.add_line(entity.dxf.start, entity.dxf.end, dxfattribs=entity.dxfattribs())
            new_doc.saveas(file_path)
            QMessageBox.information(self, "Saved", f"File saved to {file_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dxf_path = 'test5_topleft.dxf'  # 기본 DXF 파일 경로
    viewer = DXFViewer(dxf_path)
    viewer.show()
    sys.exit(app.exec_())
