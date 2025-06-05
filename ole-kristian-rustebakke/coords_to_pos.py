import numpy as np 


def increase_and_constant(point_1, point_2):
    a = (point_2[1] - point_1[1])/(point_2[0] - point_1[0])
    b = point_1[1] - a*point_1[0]

    return a, b  

def calc_lines_and_diag(P1, P2, P3, P4):
    l1 = increase_and_constant(P1, P3)
    l2 = increase_and_constant(P2, P4)
    l3 = increase_and_constant(P1, P2)
    l4 = increase_and_constant(P3, P4)

    d1 = increase_and_constant(P1, P4)
    d2 = increase_and_constant(P2, P3)

    return l1, l2, l3, l4, d1, d2

def find_corner_points(l1, l2, l3, l4): 
    P1 = find_intersection(l1, l3)
    P2 = find_intersection(l2, l3)
    P3 = find_intersection(l1, l4)
    P4 = find_intersection(l2, l4)

    return P1, P2, P3, P4

def find_cross_midpoints_and_midlines(l1, l2, l3, l4, d1, d2):
    CP1 = find_intersection(d1, d2)
    # CP1_x = (d2[1]-d1[1])/(d1[0]-d2[0])
    # CP1_y = d1[0]*CP1_x + d1[1]
    # CP1 = (CP1_x, CP1_y)

    CP2 = find_intersection(l1, l2)
    # CP2_x = (l2[1]-l1[1])/(l1[0]-l2[0])
    # CP2_y = l1[0]*CP2_x + l1[1]
    # CP2 = (CP2_x, CP2_y)

    CP3 = find_intersection(l3, l4)
    # CP3_x = (l4[1]-l3[1])/(l3[0]-l4[0])
    # CP3_y = l3[0]*CP3_x + l3[1]
    # CP3 = (CP3_x, CP3_y)

    m1 = increase_and_constant(CP1, CP2)
    m2 = increase_and_constant(CP1, CP3)

    MP1 = find_intersection(l1, m2) 
    #MP1_x(m2[1] - l1[1])/(l1[0] - m2[0])
    #MP1_y = l1[0]*MP1_x + l1[1]
    #MP1 = (MP1_x, MP1_y)

    MP2 = find_intersection(l3, m1)
    # MP2_x = (m1[1] - l3[1])/(l3[0] - m1[0])
    # MP2_y = l3[0]*MP2_x + l3[1]
    # MP2 = (MP2_x, MP2_y)

    MP3 = find_intersection(l2, m2)
    # MP3_x = (m2[1] - l2[1])/(l2[0] - m2[0])
    # MP3_y = l2[0]*MP3_x + l2[1]
    # MP3 = (MP3_x, MP3_y)

    MP4 = find_intersection(l4, m1)
    # MP4_x = (m1[1] - l4[1])/(l4[0] - m1[0])
    # MP4_y = l4[0]*MP4_x + l4[1]
    # MP4 = (MP4_x, MP4_y)

    return CP1, CP2, CP3, MP1, MP2, MP3, MP4, m1, m2

def find_angle(point_1, point_2, point_3): #point_2 is the middle point
    vec_1 = np.array(point_1 - point_2) 
    vec_2 = np.array(point_3 - point_2) 

    cos_angle = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure it stays within valid bounds
    return np.arccos(cos_angle)

def find_quadrant(UP, P1, P2, P3, P4, CP1, CP2, CP3):
    grid_angle_y = find_angle(P1, CP2, P2)
    grid_angle_x = find_angle(P1, CP3, P3)
    if find_angle(UP, CP2, P2) > grid_angle_y and find_angle(UP, CP2, P2) > find_angle(UP, CP2, P1):
        C = 0

    if find_angle(UP, CP2, P1) < find_angle(P1, CP2, CP1) and find_angle(UP, CP2, CP1) < find_angle(P1, CP2, CP1):
        C = 1

    if find_angle(UP, CP2, P2) < find_angle(CP1, CP2, P2) and find_angle(UP, CP2, CP1) < find_angle(CP1, CP2, P2):
        C = 2

    if find_angle(UP, CP2, P1) > grid_angle_y and find_angle(UP, CP2, P1) > find_angle(UP, CP2, P2):
        C = 3

    if find_angle(UP, CP3, P3) > grid_angle_x and find_angle(UP, CP3, P3) > find_angle(UP, CP3, P1):
        R = 0

    if find_angle(UP, CP3, P1) < find_angle(P1, CP3, CP1) and find_angle(UP, CP3, CP1) < find_angle(P1, CP3, CP1):
        R = 1

    if find_angle(UP, CP3, P3) < find_angle(CP1, CP3, P3) and find_angle(UP, CP3, CP1) < find_angle(CP1, CP3, P3):
        R = 2

    if find_angle(UP, CP3, P1) > grid_angle_y and find_angle(UP, CP3, P1) > find_angle(UP, CP3, P3):
        R = 3

    return R, C #Row, Column

def find_intersection(line_1, line_2):
    x = (line_2[1] - line_1[1])/(line_1[0] - line_2[0])
    y = line_1[0]*x + line_1[1]

    return np.array([x, y])

def find_pos(UP, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2):
    position_list = [[["A", "A"],["B", "A"],["C", "A"],["D", "A"]], 
                    [["A", "B"],["B", "B"],["C", "B"],["D", "B"]], 
                    [["A", "C"],["B", "C"],["C", "C"],["D", "C"]], 
                    [["A", "D"],["B", "D"],["C", "D"],["D", "D"]]]
    x_rel_pos = position_list[R][C][0]
    y_rel_pos = position_list[R][C][1]


    x_dir = increase_and_constant(UP, P4)
    y_dir = increase_and_constant(UP, P3)
    
    IP1_x = find_intersection(x_dir, l1)
    IP2_x = find_intersection(x_dir, m1)
    IP3_x = find_intersection(x_dir, l2)

    #print(UP, IP1_x, IP2_x, IP3_x, x_len, y_len)

    IP1_y = find_intersection(y_dir, l3)
    IP2_y = find_intersection(y_dir, m2)
    IP3_y = find_intersection(y_dir, l4)
    #print(UP, IP1_y, IP2_y, IP3_y, x_len, y_len)

    if x_rel_pos == "A":
        c = (np.linalg.norm(IP2_x - UP)*np.linalg.norm(IP3_x - IP1_x))/(np.linalg.norm(IP2_x - IP1_x)*np.linalg.norm(IP3_x - UP))
        #print(c)
        x = x_len/(2 - c)*(1 - c)

    if y_rel_pos == "A":
        c = (np.linalg.norm(IP2_y - UP)*np.linalg.norm(IP3_y - IP1_y))/(np.linalg.norm(IP2_y - IP1_y)*np.linalg.norm(IP3_y - UP))
        #print(c)
        y = y_len/(2 - c)*(1 - c)

    if x_rel_pos == "B":
        c = (np.linalg.norm(IP2_x - IP1_x)*np.linalg.norm(IP3_x - UP))/(np.linalg.norm(IP2_x - UP)*np.linalg.norm(IP3_x - IP1_x))
        x = x_len/(1 - 2*c)*(1 - c)

    if y_rel_pos == "B":
        c = (np.linalg.norm(IP2_y - IP1_y)*np.linalg.norm(IP3_y - UP))/(np.linalg.norm(IP2_y - UP)*np.linalg.norm(IP3_y - IP1_y))
        y = y_len/(1 - 2*c)*(1 - c)

    if x_rel_pos == "C":
        c = (np.linalg.norm(UP - IP1_x)*np.linalg.norm(IP3_x - IP2_x))/(np.linalg.norm(UP - IP2_x)*np.linalg.norm(IP3_x - IP1_x))
        x = c*x_len/(2*c - 1)

    if y_rel_pos == "C":
        c = (np.linalg.norm(UP - IP1_y)*np.linalg.norm(IP3_y - IP2_y))/(np.linalg.norm(UP - IP2_y)*np.linalg.norm(IP3_y - IP1_y))
        y = c*y_len/(2*c - 1)

    if x_rel_pos == "D": 
        c = (np.linalg.norm(IP3_x - IP1_x)*np.linalg.norm(UP - IP2_x))/(np.linalg.norm(IP3_x - IP2_x)*np.linalg.norm(UP - IP1_x))
        x = x_len/(2 - c)

    if y_rel_pos == "D": 
        c = (np.linalg.norm(IP3_y - IP1_y)*np.linalg.norm(UP - IP2_y))/(np.linalg.norm(IP3_y - IP2_y)*np.linalg.norm(UP - IP1_y))
        y = y_len/(2 - c)
    #print("x",x, "y", y)
    #print(R, C)
    return np.array([x, y])

def find_ellipse(circle_points, height, width):
    X = np.array([point['x']*height for point in circle_points])
    Y = np.array([point['y']*width for point in circle_points])

    A = np.column_stack([X**2, X * Y, Y**2, X, Y])  # Design matrix with 5 columns
    b = np.ones_like(X)  

    # Solve the least squares problem to find the coefficients of the ellipse equation
    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

    # Extract the coefficients
    A, B, C, D, E = coefficients

    return A, B, C, D, E

def find_intersect_ellipse_line(A, B, C, D, E, line):
    a = A + B*line[0] + C*line[0]**2
    b = B*line[1] + 2*C*line[0]*line[1] + D + E*line[0]
    c = C*line[1]**2 + line[1] - 1 

    x_1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    x_2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return np.array([x_1, x_2])

def find_ellipse_tangent(A, B, C, D, E, point):
    a = -(2*A*point[0] + B*point[1] + D)/(B*point[0] + 2*C*point[1] + E)
    b = point[1] - a*point[0]

    return a, b
# P1 = np.array([-587.3, 124.4])
# P2 = np.array([-109.5, 76.6])
# P3 = np.array([-368.2, 170.5])
# P4 = np.array([71.8, 132.6])
# UP = np.array([-61.7, 112.9])
# x_len = 16.5
# y_len = 10.96


# l1, l2, l3, l4, d1, d2 = calc_lines_and_diag(P1, P2, P3, P4)
# CP1, CP2, CP3, MP1, MP2, MP3, MP4, m1, m2 = find_cross_midpoints_and_midlines(l1, l2, l3, l4, d1, d2)
# R, C = find_quadrant(UP, P1, P2, P3, P4, CP1, CP2, CP3)

# import matplotlib.pyplot as plt

# x_dir = increase_and_constant(UP, P3)
# y_dir = increase_and_constant(UP, P4)

# IP1_x = find_intersection(x_dir, l1)
# IP2_x = find_intersection(x_dir, m1)
# IP3_x = find_intersection(x_dir, l2)

# IP1_y = find_intersection(y_dir, l3)
# IP2_y = find_intersection(y_dir, m2)
# IP3_y = find_intersection(y_dir, l4)

# x_vals = np.linspace(-600, 100, 400)

# # Compute y-values for each line using y = mx + b
# y_l1 = l1[0] * x_vals + l1[1]
# y_l2 = l2[0] * x_vals + l2[1]
# y_l3 = l3[0] * x_vals + l3[1]
# y_l4 = l4[0] * x_vals + l4[1]
# y_m1 = m1[0] * x_vals + m1[1]
# y_m2 = m2[0] * x_vals + m2[1]

# y_x_dir = x_dir[0]*x_vals + x_dir[1]
# y_y_dir = y_dir[0]*x_vals + y_dir[1]
# # Create the plot
# plt.figure(figsize=(8, 6))

# # Plot each line
# plt.plot(x_vals, y_l1, label='Line l1', color='blue')
# plt.plot(x_vals, y_l2, label='Line l2', color='green')
# plt.plot(x_vals, y_l3, label='Line l3', color='red')
# plt.plot(x_vals, y_l4, label='Line l4', color='purple')
# plt.plot(x_vals, y_m1, label='Midline m1', color='orange')
# plt.plot(x_vals, y_m2, label='Midline m2', color='brown')

# plt.plot(x_vals, y_x_dir, label='y_x_dir', color='indigo')
# plt.plot(x_vals, y_y_dir, label='y_y_dir', color='black')


# points = np.array([P1, P2, P3, P4, CP1, MP1, MP2, MP3, MP4, UP, IP1_x, IP2_x, IP3_x, IP1_y, IP2_y, IP3_y])  # Create a numpy array for easy plotting
# plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label="Points")

# # Add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Plot of Lines l1, l2, l3, l4, m1, m2')

# # Display legend
# plt.legend()

# # Show the plot
# plt.grid(True)
# plt.savefig("Testgrid.png")

# print(find_pos(UP, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2))