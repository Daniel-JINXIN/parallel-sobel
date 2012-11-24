#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "max_subarray.hpp"

using namespace std;


void print_matrix(vector<vector<int>> mat);
vector<vector<int>> parseFile(string fileName);
Matrix computeCumulMatrix(const Matrix& mat);



Matrix::Matrix() {}
Matrix::~Matrix() {}



ComputedMatrix::ComputedMatrix(std::vector<std::vector<int>> _data)
        : Matrix(_data)
{
        cumulMatrix = computeCumulMatrix(*this);
}


ComputedMatrix::~ComputedMatrix() {}







SubMatrix ComputedMatrix::kandane(int startLine, int endLine) const
{
        vector<int> line(data[endLine]);

        if (startLine != 0) {
                for (unsigned int i = 0; i < line.size(); i++) {
                        line[i] -= data[startLine-1][i];
                }
        }

        return kandane(line);
}




SubMatrix ComputedMatrix::kandane(vector<int> line)
{
        SubMatrix max;
        int tmp_max_sum = 0;
        int tmp_start = 0;
        max.startX = 0;
        max.endX = 0;
        max.startY = 0;
        max.endY = 0;
        max.sum = -32001;

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

        print_matrix(matAsVect);
        cout << endl << endl;
        print_matrix(mat.getData());
        cout << endl << endl;
        print_matrix(mat.getCumulMatrix().getData());

        //SubMatrix maxSubarray = mat.maxSubarray();

        SubMatrix max = ComputedMatrix::kandane(matAsVect[0]);
        cout << max.toString() << endl;

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





//// #define _PRINT_INFO 

//long get_usecs(void)
//{
	//struct timeval t;
	//gettimeofday(&t, NULL);
	//return t.tv_sec * 1000000 + t.tv_usec;
//}

//void usage(const char *app_name)
//{
	//printf("Argument error! Usage: %s <input_file> \n", app_name);
	//exit(0);
//}



//void clear(int *a, int len)
//{
	//for (int index = 0; index < len; index++) {
		//*(a + index) = 0;
	//}
//}






//parse_file()
//{
	//// Read the matrix
	//int dim = 0;
	//fscanf(input_file, "%u\n", &dim);
	//int mat[dim][dim];
	//int element = 0;
	//for (int i = 0; i < dim; i++) {
		//for (int j = 0; j < dim; j++) {
			//if (j != (dim - 1))
				//fscanf(input_file, "%d\t", &element);
			//else
				//fscanf(input_file, "%d\n", &element);
			//mat[i][j] = element;
		//}
	//}
//}









//int main(int argc, char *argv[])
//{
	//if (argc != 2) {
		//usage(argv[0]);
	//}
	//// Open files
	//FILE *input_file = fopen(argv[1], "r");
	//if (input_file == NULL) {
		//usage(argv[0]);
	//}

	//// Algorithm based on information obtained here:
	//// http://stackoverflow.com/questions/2643908/getting-the-submatrix-with-maximum-sum
	//long alg_start = get_usecs();
	//// Compute vertical prefix sum
	//int ps[dim][dim];
	//[>for (int i=0; i<dim; i++) {
	   //for (int j=0; j<dim; j++) {
	   //if (j == 0) {
	   //ps[j][i] = mat[j][i];
	   //} 
	   //else {
	   //ps[j][i] = mat[j][j] + ps[i-1][i];
	   //}
	   //}
	   //} */
	//for (int j = 0; j < dim; j++) {
		//ps[0][j] = mat[0][j];
		//for (int i = 1; i < dim; i++) {
			//ps[i][j] = ps[i - 1][j] + mat[i][j];
		//}
	//}

//#ifdef _PRINT_INFO
	//// Print the matrix
	//printf("Vertical prefix sum matrix [%d]\n", dim);
	//for (int i = 0; i < dim; i++) {
		//for (int j = 0; j < dim; j++) {
			//printf("%d\t", ps[i][j]);
		//}
		//printf("\n");
	//}
//#endif

	//int max_sum = mat[0][0];
	//int top = 0, left = 0, bottom = 0, right = 0;

	////Auxilliary variables 
	//int sum[dim];
	//int pos[dim];
	//int local_max;

	//for (int i = 0; i < dim; i++) {
		//for (int k = i; k < dim; k++) {
			//// Kandane over all columns with the i..k rows
			//clear(sum, dim);
			//clear(pos, dim);
			//local_max = 0;

			//// We keep track of the position of the max value over each Kandane's execution
			//// Notice that we do not keep track of the max value, but only its position
			//sum[0] = ps[k][0] - (i == 0 ? 0 : ps[i - 1][0]);
			//for (int j = 1; j < dim; j++) {
				//if (sum[j - 1] > 0) {
					//sum[j] = sum[j - 1] + ps[k][j] - (i == 0 ? 0 : ps[i - 1][j]);
					//pos[j] = pos[j - 1];
				//} else {
					//sum[j] = ps[k][j] - (i == 0 ? 0 : ps[i - 1][j]);
					//pos[j] = j;
				//}
				//if (sum[j] > sum[local_max]) {
					//local_max = j;
				//}
			//}	//Kandane ends here

			//if (sum[local_max] > max_sum) {
				//// sum[local_max] is the new max value
				//// the corresponding submatrix goes from rows i..k.
				//// and from columns pos[local_max]..local_max

				//max_sum = sum[local_max];
				//top = i;
				//left = pos[local_max];
				//bottom = k;
				//right = local_max;
			//}
		//}
	//}

	//// Compose the output matrix
	//int outmat_row_dim = bottom - top + 1;
	//int outmat_col_dim = right - left + 1;
	//int outmat[outmat_row_dim][outmat_col_dim];
	//for (int i = top, k = 0; i <= bottom; i++, k++) {
		//for (int j = left, l = 0; j <= right; j++, l++) {
			//outmat[k][l] = mat[i][j];
		//}
	//}
	//long alg_end = get_usecs();

	//// Print output matrix
        //printf("Sub-matrix [%dX%d] with max sum = %d, top = %d, bottom = %d, left = %d, right = %d\n",
             //outmat_row_dim, outmat_col_dim, max_sum, top, bottom, left, right);
        //[>printf("C: Max value: %d, between (%d, %d) and (%d, %d)\n", max_sum, left, top, right, bottom);<]
//#ifdef _PRINT_INFO
	//for (int i = 0; i < outmat_row_dim; i++) {
		//for (int j = 0; j < outmat_col_dim; j++) {
			//printf("%d\t", outmat[i][j]);
		//}
		//printf("\n");
	//}
//#endif

	//// Release resources
	//fclose(input_file);

	//// Print stats
	//[>printf("%s,arg(%s),%s,%f sec\n", argv[0], argv[1], "CHECK_NOT_PERFORMED", ((double)(alg_end - alg_start)) / 1000000);<]
	//return 0;
//}
