/////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Written by Cyril Cousinou and Yoann Ricordel ///////////////////////////
////////////////////////////// Final version submitted on the 12/03/12 //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __MAX_SUBARRAY_H__
#define __MAX_SUBARRAY_H__

#include <vector>


class SubMatrix {
        public:
                int startX;
                int endX;
                int startY;
                int endY;
                int sum;

                SubMatrix();
                SubMatrix(int _startX, int _endX, int _startY, int _endY, int _sum);
				
				
                std::string toString();
};


class Matrix
{
        public:
                Matrix();
                Matrix(std::vector<std::vector<int>> _data);
                Matrix(const Matrix& m);
                ~Matrix();

                inline std::vector<std::vector<int>> getData() { return data; }
                inline int getWidth() const { return width; }
                inline int getHeight() const { return height; }
                inline int getDataAt(int i, int j) const { return data[i][j]; }
                inline void setDataAt(int i, int j, int val) { data[i][j] = val; }

                inline std::vector<int>& operator[] (unsigned int i) { return data[i]; }
        protected:
                int height;
                int width;
                std::vector<std::vector<int>> data;
};



class ComputedMatrix : public Matrix
{
        public:
                ComputedMatrix(std::vector<std::vector<int>> _data);
                ~ComputedMatrix();
                SubMatrix maxSubarray();
                inline Matrix getCumulMatrix() { return cumulMatrix; }

                static SubMatrix kandane(std::vector<int> line); //XXX should be private

        private:
                Matrix cumulMatrix;

                /* Computes the max subarray that spans between startLine and endLine */
                SubMatrix kandane(int startLine, int endLine);
};



std::vector<std::vector<int>> parseFile(std::string filename);




#endif /* end of include guard: __MAX_SUBARRAY_H__ */
