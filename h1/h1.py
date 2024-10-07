import pathlib
import math


# 1
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    matrix_file = []
    for line in path.read_text().splitlines():
        previous_word = ""
        for word in line.split(" "):
            if word != '+' and word != '-' and word != '=':
                if word.__contains__('x') == True or word.__contains__('y') == True or word.__contains__('z') == True:
                    if len(word) == 1:
                        if previous_word == "-":
                            matrix_file.append(float(-1))
                        else:
                            matrix_file.append(float(1))
                    else:
                        if previous_word == "-":
                            matrix_file.append(-float(word[:len(word) - 1]))
                        else:
                            matrix_file.append(float(word[:len(word) - 1]))
                else:
                    if previous_word == "-":
                        matrix_file.append(-float(word))
                    else:
                        matrix_file.append(float(word))
            previous_word = word
    matrix_A = [[matrix_file[0], matrix_file[1], matrix_file[2]], [matrix_file[4], matrix_file[5], matrix_file[6]],
                [matrix_file[8], matrix_file[9], matrix_file[10]]]
    matrix_B = [matrix_file[3], matrix_file[7], matrix_file[11]]

    return matrix_A, matrix_B


A, B = load_system(pathlib.Path("h1/system.txt"))
print(f"{A=} {B=}")


# 2
# 2.1
def determinant(matrix: list[list[float]]) -> float:
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    return det


print(f"{determinant(A)=}")


# 2.2
def trace(matrix: list[list[float]]) -> float:
    tr = matrix[0][0] + matrix[1][1] + matrix[2][2]
    return tr


print(f"{trace(A)=}")


# 2.3
def norm(vector: list[float]) -> float:
    n = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    return n


print(f"{norm(B)=}")


# 2.4
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    matrix_transposed = [[], [], []]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix_transposed[i].append(matrix[j][i])
    return matrix_transposed


print(f"{transpose(A)=}")


# 2.5
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = [0, 0, 0]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result[i] += matrix[i][j] * vector[j]
    return result


print(f"{multiply(A, B)=}")


# 3
def replace_column(matrix: list[list[float]], vector: list[float], poz) -> list[list[float]]:
    new_matrix = [[0.0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    if 0 <= poz <= 3:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_matrix[i][j] = matrix[i][j]
        for i in range(len(matrix)):
            new_matrix[i][poz] = vector[i]
    return new_matrix


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_matrix = determinant(matrix)
    if det_matrix == 0:
        raise ValueError("determinant = 0")

    solved_matrix = [0] * len(vector)
    for i in range(0, len(matrix)):
        solved_matrix[i] = determinant(replace_column(matrix, vector, i)) / det_matrix
    return solved_matrix


print(f"{solve_cramer(A, B)=}")


# 4
def determinant_2(matrix: list[list[float]]) -> float:
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return det


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minor_matrix = []
    for z in range(len(matrix)):
        new_line = []
        for k in range(len(matrix[z])):
            if z != i and k != j:
               new_line.append(matrix[z][k])
        if new_line != []:
            minor_matrix.append(new_line)
    return minor_matrix


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactor_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cofactor_matrix[i][j] = math.pow(-1, i + j) * determinant_2(minor(matrix, i, j))
    return cofactor_matrix


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    inverse_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    adj_matrix = adjoint(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            inverse_matrix[i][j] = (1 / determinant(matrix)) * adj_matrix[i][j]
    return inverse_matrix


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return multiply(inverse(matrix), vector)


print(f"{solve(A, B)=}")
