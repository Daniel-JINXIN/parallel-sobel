/////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Written by Cyril Cousinou and Yoann Ricordel ///////////////////////////
////////////////////////////// Final version submitted on the 12/03/12 //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

/*Include all the required librairies*/
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <omp.h>

#include "max_subarray.hpp"

using namespace std;

/*Declarations of functions*/
void print_matrix(vector<vector<int>> mat);
vector<vector<int>> parseFile(string fileName);
Matrix computeCumulMatrix(const Matrix& mat);
std::vector<int> cumulLine(const std::vector<int>& line);

/*Constructor and destructor of the Matrix class*/
Matrix::Matrix() {}
Matrix::~Matrix() {}


SubMatrix::SubMatrix()
		: startX(0), endX(0), startY(0), endY(0), sum(-32001) {}

SubMatrix::SubMatrix(int _startX, int _endX, int _startY, int _endY, int _sum)
        : startX(_startX), endX(_endX), startY(_startY), endY(_endY), sum(_sum) {}

Matrix::Matrix(vector<vector<int>> _data)
        : height(_data.size()), width(_data[1].size()), data(_data)
{}

Matrix::Matrix(const Matrix& m)
        : height(m.height), width(m.width), data(m.data)
{
}
		
/*Constructor and destructor of the ComputedMatrix class*/
ComputedMatrix::ComputedMatrix(std::vector<std::vector<int>> _data)
        : Matrix(_data)
{
        cumulMatrix = computeCumulMatrix(*this);
}

ComputedMatrix::~ComputedMatrix() {}


std::string SubMatrix::toString()
{
        std::stringstream s_str;

        s_str << startX << " " << startY << " " << endX << " " << endY << endl;
        return s_str.str();
}


SubMatrix ComputedMatrix::maxSubarray()
{
		std::vector<SubMatrix> max;
		
		#pragma omp parallel shared(max)
		{
			int nb_threads = omp_get_num_threads();
			
			/* We don't want the vector to be resized by each thread */
			#pragma omp single
				max.resize(nb_threads);
				
			int workload = height/nb_threads;
			
			#pragma omp for schedule(static, workload)
				for (int i = 0; i < height; i++) {
					for (int j = i; j < height; j++) {
							 
							SubMatrix local_max = kandane(i, j);
							
							int id = omp_get_thread_num();
							
							/* Update the maximum sum, local to the thread, if relevant */
							if (local_max.sum > max[id].sum) {	
								max[id] = local_max;
							}
					}
				}
		}
		
		SubMatrix global_max(0, 0, 0, 0, -32001);
		
		/* Select the maximum subarray between the local ones of each thread */
		for (unsigned int k = 0; k < max.size(); k++) {
			if (max[k].sum > global_max.sum) {
				global_max = max[k];
			}						
		}

        return global_max;
}


SubMatrix ComputedMatrix::kandane(int startLine, int endLine)
{
        vector<int> line(cumulMatrix[endLine]);
		
		/* Computes the relevant line in which to search a maximum subarray */
		/* with help of the derivative matrix */
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
		
		/* Computes the sum of all subarrays of a line while selecting the maximum one */
		/* If a sum becomes negative, we switch to the next subarray */
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


Matrix computeCumulMatrix(const Matrix& mat)
{
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
        /*Check the arguments are correctly entered*/
		if (argc < 2) {
                printf("Usage: %s fileName\n", argv[0]);
                exit(1);
        }
		
		/* Loop on the files entered as arguments */	
		for (int i = 1; i < argc; i++) {
		
			/*Parse the file, throwing an exception if the file does not exist*/
			string fileName = argv[i];
			vector<vector<int>> matAsVect;
			try {
					matAsVect = parseFile(fileName);
			} catch (const std::exception &ex) {
					cout << "Could not parse file " << fileName << endl;
					throw;
			}
			
			/* Build an object from the parsed matrix on which to apply the kandane algorithm*/
			ComputedMatrix mat(matAsVect);

			/* Find the maximum subarray of the original matrix...*/
			SubMatrix maxSubarray = mat.maxSubarray();
			
			/*...and print it out on the console*/
			cout << maxSubarray.toString();
		}
				
        return 0;
}


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
