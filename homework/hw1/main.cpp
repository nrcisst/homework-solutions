#include  <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>

using namespace std;

vector<vector<int>> readMatrix(const string& filename) {
    vector<vector<int>> matrix;
    ifstream in(filename);
    string line;

    if (!in) {
        cerr << "Cannot open the File : " << filename << endl;
        return matrix;
    }

    while (getline(in, line)) {
        vector<int> row;
        stringstream ss(line);
        int value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }
    in.close();
    return matrix;
}

int calculateMinimumPathSum(vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;

    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<int>> dp(rows, vector<int>(cols, 0));

    dp[0][0] = matrix[0][0];

    for (int i = 1; i < rows; ++i) {
        dp[i][0] = dp[i - 1][0] + matrix[i][0];
    }

    for (int j = 1; j < cols; ++j) {
        dp[0][j] = dp[0][j - 1] + matrix[0][j];
    }

    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
        }
    }

    return dp[rows - 1][cols - 1];
}

int main(int argc, char * argv[]) {
    string filename = argv[1]; // The file should be in the same folder as your executable
    vector<vector<int>> matrix = readMatrix(filename);

    int result = calculateMinimumPathSum(matrix);
    cout << "The minimum path sum from the top left to the bottom right of the matrix is: " << result << endl;

    return 0;
}
