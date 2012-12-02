/////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Written by Cyril COusinou and Yoann Ricordel ///////////////////////////
////////////////////////////// Final version submitted on the 12/03/12 //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <omp.h>

#include "max_subarray.hpp"

using namespace std;


void print_matrix(vector<vector<int>> mat);
vector<vector<int>> parseFile(string fileName);
Matrix computeCumulMatrix(const Matrix& mat);
std::vector<int> cumulLine(const std::vector<int>& line);


Matrix::Matrix() {}
Matrix::~Matrix() {}


SubMatrix::SubMatrix()
		: startX(0), endX(0), startY(0), endY(0), sum(-32001) {}

SubMatrix::SubMatrix(int _startX, int _endX, int _startY, int _endY, int _sum)
        : startX(_startX), endX(_endX), startY(_startY), endY(_endY), sum(_sum) {}


ComputedMatrix::~ComputedMatrix() {}


ComputedMatrix::ComputedMatrix(std::vector<std::vector<int>> _data)
        : Matrix(_data)
{
        cumulMatrix = computeCumulMatrix(*this);
}


Matrix::Matrix(vector<vector<int>> _data)
        : height(_data.size()), width(_data[1].size()), data(_data)
{}

Matrix::Matrix(const Matrix& m)
        : height(m.height), width(m.width), data(m.data)
{
}


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
				
			#pragma omp single
				max.resize(nb_threads);
				
			int workload = height/nb_threads;
				
			#pragma omp for schedule(static, workload)
				for (int i = 0; i < height; i++) {
					for (int j = i; j < height; j++) {
							SubMatrix local_max = kandane(i, j);
							
							int id = omp_get_thread_num();
							
							if (local_max.sum > max[id].sum) {	
								max[id] = local_max;
							}
					}
				}
		}
		
		SubMatrix global_max(0, 0, 0, 0, -32001);
		
		for (unsigned int k = 0; k < max.size(); k++) {
			if (max[k].sum > global_max.sum) {
				global_max = max[k];
			}						
		}

        return global_max;
}


void print_line(std::vector<int> line)
{
        for (unsigned int i = 0; i < line.size(); i++) {
                cout << line[i] << " ";
        }
		cout << endl;
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
        if (argc < 2) {
                printf("Usage: %s fileName\n", argv[0]);
                exit(1);
        }
				
		for (int i = 1; i < argc; i++) {
			string fileName = argv[i];
			vector<vector<int>> matAsVect;
			try {
					matAsVect = parseFile(fileName);
			} catch (const std::exception &ex) {
					cout << "Could not parse file " << fileName << endl;
					throw;
			}
					
			/*Beginning of work to be parallelized*/
			double timer1 = 0;
			timer1 = omp_get_wtime();
			
			ComputedMatrix mat(matAsVect);

			SubMatrix maxSubarray = mat.maxSubarray();
			/*End of work to be parallelized*/

			double timer2 = 0;
			timer2 = omp_get_wtime();
                        {
                                int numThreads = atoi(getenv("OMP_NUM_THREADS"));
                                double t = timer2 - timer1;
                                //printf("Elapsed time was: %f\n", timer2-timer1);
                                int size = mat.getHeight() * mat.getWidth();
                                fprintf(stdout, "{\"name\": \"%s\", \"size\": %u, \"nProcs\": %u, \"time\": %lf, \"throughput\": %lf},\n",
                                                "max subarray laptop", size, numThreads, t, (double)size/t);
                        }
			
			cout << maxSubarray.toString();

		}
				
        return 0;
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
