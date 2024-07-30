import sys
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QListWidget, QHBoxLayout, \
    QMessageBox, QMenuBar, QAction, QFileDialog


class DXFViewer(QMainWindow):
    def __init__(self, dxf_path=None):
        super().__init__()
        self.setWindowTitle('DXF Viewer')
        self.setGeometry(100, 100, 800, 600)

        self.dxf_path = dxf_path
        self.doc = None
        self.msp = None
        self.entities = []

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

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()

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

        layout.addLayout(button_layout)

    def load_dxf(self, file_path):
        self.doc = ezdxf.readfile(file_path)
        self.msp = self.doc.modelspace()
        self.entities = list(self.msp.query('LWPOLYLINE LINE CIRCLE'))
        self.update_list_widget()
        self.visualize_entities()

    def update_list_widget(self):
        self.list_widget.clear()
        for i, entity in enumerate(self.entities):
            layer_name = entity.dxf.layer
            entity_type = entity.dxftype()
            self.list_widget.addItem(f'{entity_type} {i + 1} (Layer: {layer_name})')

    def move_up(self):
        current_row = self.list_widget.currentRow()
        if current_row > 0:
            current_item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row - 1, current_item)
            self.list_widget.setCurrentRow(current_row - 1)

    def move_down(self):
        current_row = self.list_widget.currentRow()
        if current_row < self.list_widget.count() - 1:
            current_item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row + 1, current_item)
            self.list_widget.setCurrentRow(current_row + 1)

    def delete_selected(self):
        current_row = self.list_widget.currentRow()
        if current_row != -1:
            del self.entities[current_row]
            self.list_widget.takeItem(current_row)
            self.update_list_widget()
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
        order = [int(self.list_widget.item(i).text().split()[1]) - 1 for i in range(self.list_widget.count())]
        self.entities = [self.entities[i] for i in order]
        self.update_list_widget()
        self.visualize_entities()

    def visualize_entities(self):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        for idx, entity in enumerate(self.entities):
            if entity.dxftype() == 'LWPOLYLINE':
                vertices = entity.get_points('xy')
                x, y = zip(*vertices)
                ax.plot(x, y)
                for i in range(len(vertices) - 1):
                    start = vertices[i]
                    end = vertices[i + 1]
                    ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                             head_width=2, head_length=2, fc='red', ec='red')
                start_x, start_y = vertices[0]
                ax.text(start_x, start_y, str(idx + 1), fontsize=12, color='blue', ha='right')
            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                ax.plot([start.x, end.x], [start.y, end.y])
                ax.arrow(start.x, start.y, end.x - start.x, end.y - start.y,
                         head_width=2, head_length=2, fc='red', ec='red')
                ax.text(start.x, start.y, str(idx + 1), fontsize=12, color='blue', ha='right')
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                circle = plt.Circle((center.x, center.y), radius, edgecolor='blue', facecolor='none')
                ax.add_artist(circle)
                ax.text(center.x, center.y, str(idx + 1), fontsize=12, color='blue', ha='center', va='center')

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
                elif entity.dxftype() == 'CIRCLE':
                    new_msp.add_circle(entity.dxf.center, entity.dxf.radius, dxfattribs=entity.dxfattribs())
            new_doc.saveas(file_path)
            QMessageBox.information(self, "Saved", f"File saved to {file_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # dxf_path = 'dxf_examples/e66s_21.50mm_12.64mm_20240127_0차.dxf'  # 기본 DXF 파일 경로
    dxf_path = 'test3.dxf'  # 기본 DXF 파일 경로
    viewer = DXFViewer(dxf_path)
    viewer.show()
    sys.exit(app.exec_())
