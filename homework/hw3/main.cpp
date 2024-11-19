#include <fstream>
#include <iostream>
#include <vector>
#include "Stack.h"
#include "Queue.h"
#include "ArgumentManager.h"

using namespace std;

struct Position {
  int x, y;
  Position() : x(0), y(0) {}
  Position(int xPos, int yPos) : x(xPos), y(yPos) {}
};

const int dX[8] = {2, 1, -1, -2, -2, -1, 1, 2};
const int dY[8] = {1, 2, 2, 1, -1, -2, -2, -1};

bool IsSafe(int x, int y, int numRows, int numCols, const vector<vector<int>>& board) {
  return x >= 0 && y >= 0 && x < numRows && y < numCols && board[x][y] == 0;
}

bool SolveKTUtil(int x, int y, int moveI, vector<vector<int>>& board, int numRows, int numCols, Stack<Position>& path, const Position& start) {\
  if (moveI == numRows * numCols) {
    for (int k = 0; k < 8; k++) {
      int nextX = x + dX[k];
      int nextY = y + dY[k];
      if (nextX == start.x && nextY == start.y) return true;
    }
    return false;
  }

  for (int k = 0; k < 8; k++) {
    int nextX = x + dX[k];
    int nextY = y + dY[k];
    if (IsSafe(nextX, nextY, numRows, numCols, board)) {
      board[nextX][nextY] = moveI;
      path.Push(Position(nextX, nextY));
      if (SolveKTUtil(nextX, nextY, moveI + 1, board, numRows, numCols, path, start)) return true;
      else {
        board[nextX][nextY] = 0;
        path.Pop();
      }
    }
  }
  return false;
}

bool SolveKnightTour(int numRows, int numCols, const Position& start, const vector<Position>& holes, const string& output) {
  vector<vector<int>> board(numRows, vector<int>(numCols, 0));
  Stack<Position> path;

  ofstream out(output);
  for (const auto& hole : holes) {
    board[hole.x][hole.y] = -1;
  }
  
  board[start.x][start.y] = 1;
  path.Push(start);

  if (!SolveKTUtil(start.x, start.y, 2, board, numRows, numCols, path, start)) {
    out << "A closed tour does not exist!" << endl;
    return false;
  } else {
    out << "A closed tour exists!" << endl;
    for (int i = 0; i < numRows; i++) {
      out << "+--+--+--+" << endl;
      for (int j = 0; j < numCols; j++) {
        if (board[i][j] == -1) out << "|XX";
        else {
          out << "|";
          if (board[i][j] < 10) out << " ";
          out << board[i][j];
        }
      }
      out << "|" << endl;
    }
    out << "+--+--+--+" << endl;
    return true;
  }
}

int main(int argc, char  * argv[]) {
    ArgumentManager am(argc, argv);
    string in = am.get("input");
    string output =am.get("output");
    
    ifstream input(in);
    ofstream out(output);
  if (!input) {
    cout << "Error: Unable to open file." << endl;
    return 0;
  }
  int numRows, numCols, numHoles, startX, startY;
  input >> numRows >> numCols;
  input >> startX >> startY;
  input >> numHoles;
  
  vector<Position> holes(numHoles);
  for (int i = 0; i < numHoles; i++) {
    int holeX, holeY;
    input >> holeX >> holeY;
    holes[i] = Position(holeX, holeY);
  }

  SolveKnightTour(numRows, numCols, Position(startX, startY), holes, output);

  return 0;
}
