import ezdxf
import matplotlib.pyplot as plt

# DXF 파일 읽기
file_path = 'dxf_examples/e66s_21.50mm_12.64mm_20240127_0차.dxf'
doc = ezdxf.readfile(file_path)

# 모든 폴리라인 엔티티를 가져오기
msp = doc.modelspace()
polylines = msp.query('LWPOLYLINE')

# 폴리라인의 방향을 시각화
plt.figure(figsize=(10, 10))

for pline in polylines:
    vertices = pline.get_points('xy')
    x, y = zip(*vertices)
    # plt.plot(x, y, marker='o')
    plt.plot(x, y)
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=2, head_length=2, fc='red', ec='red')

plt.title('Polyline Directions')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
