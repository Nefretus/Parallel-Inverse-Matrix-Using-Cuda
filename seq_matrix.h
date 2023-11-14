#pragma once

std::vector<std::vector<float>> generate_matrix(size_t size);
std::vector<std::vector<float>> create_identity(size_t size);
void print_matrix(std::vector<std::vector<float>>& matrix, const std::string& message);
void calculate_inverse_seq(std::vector<std::vector<float>>& input_matrix, std::vector<std::vector<float>>& identity);
