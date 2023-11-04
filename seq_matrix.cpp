#include<iostream>
#include<vector>
#include<cstdlib> 
#include<chrono>

// zrodla:
// https://cse.buffalo.edu/faculty/miller/Courses/CSE633/thanigachalam-Spring-2014-CSE633.pdf

std::vector<std::vector<float>> generate_matrix(size_t size) {
	std::vector<std::vector<float>> matrix;
	matrix.reserve(size);
	for (int i = 0; i < size; i++) {
		std::vector<float> row(size, 0);
		for (int j = 0; j < size; j++)
			row[j] = static_cast<float>((rand() % 99) + 10);
		matrix.push_back(row);
	}
	return matrix;
}

std::vector<std::vector<float>> create_identity(size_t size) {
	std::vector<std::vector<float>> matrix(size, std::vector<float>(size, 0.0));
	for (int i = 0; i < size; i++)
		matrix[i][i] = 1.0;
	return matrix;
}

void print_matrix(std::vector<std::vector<float>>& matrix, const std::string& message) {
	std::cout << std::endl << message << std::endl;
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix.size(); j++)
			std::cout << matrix[i][j] << " ";
		std::cout << std::endl;
	}
}

void calculate_inverse_seq(size_t size) {
	auto input_matrix = generate_matrix(size);
	auto identity = create_identity(size);
	//print_matrix(input_matrix, "Input matrix: ");
	for (int i = 0; i < size; i++) {
		bool inverse_exists = true;
		if (input_matrix[i][i] == 0) {
			for (int m = i + 1; m < size; m++) {
				if (input_matrix[m][i] != 0.0) {
					std::swap(input_matrix[i], input_matrix[m]);
					break;
				}
			}
			inverse_exists = false;
		}
		if (!inverse_exists) {
			std::cout << "Inverse matrix does not exists" << std::endl;
			return;	
		}
		float scale = input_matrix[i][i];
		for (int j = 0; j < size; j++) {
			input_matrix[i][j] /= scale;
			identity[i][j] /= scale;
		}
		if (i < size - 1) {
			for (int r = i + 1; r < size; r++) {
				float factor = input_matrix[r][i];
				for (int k = 0; k < size; k++) {
					input_matrix[r][k] -= factor * input_matrix[i][k];
					identity[r][k] -= factor * identity[i][k];
				}
			}
		}
	}
	for (int zeroing_col = size - 1; zeroing_col >= 1; zeroing_col--) {
		for (int row = zeroing_col - 1; row >= 0; row--) {
			float factor = input_matrix[row][zeroing_col];
			for (int col = 0; col < size; col++) {
				input_matrix[row][col] -= factor * input_matrix[zeroing_col][col];
				identity[row][col] -= factor * identity[zeroing_col][col];
			}
		}
	}
	//print_matrix(identity, "Inverse matrix: ");
}

//int main(int argv, char** argc) {
//	constexpr size_t max_size = 1000;
//	auto start = std::chrono::high_resolution_clock::now();
//	for (size_t size = 10; size <= max_size; size+=10) {
//		start = std::chrono::high_resolution_clock::now();
//		calculate_inverse_seq(size);
//		std::cout << "Size " << size << " inversion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms " << std::endl;
//	}
//	return 0;
//}

