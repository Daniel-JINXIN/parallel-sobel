#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "max_subarray.hpp"

using namespace std;


void print_matrix(vector<vector<int>> mat);
vector<vector<int>> parseFile(string fileName);
Matrix computeCumulMatrix(const Matrix& mat);


SubMatrix::SubMatrix(int _startX, int _endX, int _startY, int _endY, int _sum)
        : startX(_startX), endX(_endX), startY(_startY), endY(_endY), sum(_sum) {}


Matrix::Matrix() {}
Matrix::~Matrix() {}



ComputedMatrix::ComputedMatrix(std::vector<std::vector<int>> _data)
        : Matrix(_data)
{
        cumulMatrix = computeCumulMatrix(*this);
}


ComputedMatrix::~ComputedMatrix() {}





SubMatrix ComputedMatrix::maxSubarray()
{
        SubMatrix max(0, 0, 0, 0, -32001);
// omp parallel for ici
        for (int i = 0; i < height; i++) {
                for (int j = i; j < height; j++) {
                        SubMatrix tmp_max = kandane(i, j);
                        if (tmp_max.sum > max.sum) {
                                max = tmp_max;
                        }
                }
        }

        return max;
}



void print_line(std::vector<int> line)
{
        for (unsigned int i = 0; i < line.size(); i++) {
                cout << line[i] << " ";
        }
}



SubMatrix ComputedMatrix::kandane(int startLine, int endLine)
{
        vector<int> line(cumulMatrix[endLine]);

        if (startLine != 0) {
                for (unsigned int i = 0; i < line.size(); i++) {
                        line[i] -= cumulMatrix[startLine-1][i];
                }
        }

        SubMatrix ret = kandane(line);
        ret.startY = startLine;
        ret.endY = endLine;

        return ret;
}




SubMatrix ComputedMatrix::kandane(vector<int> line)
{
        SubMatrix max(0, 0, 0, 0, -32001);
        int tmp_max_sum = 0;
        int tmp_start = 0;

        for (unsigned int tmp_end = 0; tmp_end < line.size(); tmp_end++) {
                tmp_max_sum += line[tmp_end];
                if (tmp_max_sum > max.sum) {
                        max.sum = tmp_max_sum;
                        max.startX = tmp_start;
                        max.endX = tmp_end;
                }

                if (tmp_max_sum < 0) {
                        tmp_max_sum = 0;
                        tmp_start = tmp_end + 1;
                }
        }

        return max;
}




std::string SubMatrix::toString()
{
        std::stringstream s_str;

        s_str << "(" << startX << ", " << endX << "), (" << startY << ", " << endY << "), sum = " << sum;
        return s_str.str();
}


Matrix::Matrix(vector<vector<int>> _data)
        : height(_data.size()), width(_data[1].size()), data(_data)
{}

Matrix::Matrix(const Matrix& m)
        : height(m.height), width(m.width), data(m.data)
{
}



// Si c'est trop lent, essayer de tranposer la matrice pour la parcourir en lignes.
Matrix computeCumulMatrix(const Matrix& mat)
{
        /* Column by column for parallelism */
        int matSize = mat.getWidth();

        Matrix cumulMat(mat);

        for (int col = 0; col < matSize; col++) {
                for (int line = 1; line < matSize; line++) {
                        cumulMat.setDataAt(line, col, cumulMat.getDataAt(line-1, col) + mat.getDataAt(line, col));
                }
        }

        return cumulMat;
}






int main(int argc, const char *argv[])
{
        if (argc != 2) {
                printf("Usage: %s fileName\n", argv[0]);
                exit(1);
        }

        string fileName = argv[1];
        vector<vector<int>> matAsVect;
        try {
                matAsVect = parseFile(fileName);
        } catch (const std::exception &ex) {
                cout << "Could not parse file " << fileName << endl;
                throw;
        }


        ComputedMatrix mat(matAsVect);

        SubMatrix maxSubarray = mat.maxSubarray();
        print_matrix(mat.getData());
        cout << endl;
        cout << maxSubarray.toString() << endl;

        return 0;
}





void print_matrix(vector<vector<int>> mat)
{
        for (auto& v : mat) {
                for (auto& item : v) {
                        cout << item << " ";
                }
                cout << endl;
        }
}




/* Can throw exceptions */
vector<vector<int>> parseFile(string fileName)
{
        int matSize = 0;
        vector<vector<int>> retVect;

        ifstream matFile;
        matFile.exceptions(std::ios::failbit | std::ios::badbit);

        matFile.open(fileName);

        matFile >> matSize;

        for (int i = 0; i < matSize; i++) {
                retVect.push_back(vector<int>(matSize));
                for (int j = 0; j < matSize; j++) {
                        matFile >> retVect[i][j];
                }
        }

        return retVect;
}
