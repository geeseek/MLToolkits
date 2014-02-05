#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<set>
#include<utility>
#include<cstdlib>
#include<ctime>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

#define MAX_ROUND 10000


class splitstring : public string {
    vector<string> flds;
public:
    splitstring(char *s) : string(s) { };
    splitstring(string s) : string(s) { };
    vector<string>& split(char delim, int rep=0);
};

// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string>& splitstring::split(char delim, int rep) {
    if (!flds.empty()) flds.clear();  // empty vector if necessary
    string work = data();
    string buf = "";
    int i = 0;
    while (i < work.length()) {
        if (work[i] != delim)
            buf += work[i];
        else if (rep == 1) {
            flds.push_back(buf);
            buf = "";
        } else if (buf.length() > 0) {
            flds.push_back(buf);
            buf = "";
        }
        i++;
    }
    if (!buf.empty())
        flds.push_back(buf);
    return flds;
}

double vector_dot_product(vector<double>&x, vector<double>&y)
{
	double sum = 0.0;
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i]*y[i];
	}
	return sum;	
}

void matrix_transform(const vector<vector<double> >& matrix, vector<vector<double> >& matrixT)
{
	if (matrix.empty())
		return;
    int rowNum = matrix.size();
	int colNum = matrix[0].size();
    for (int i = 0; i < colNum; ++i)
	{
		vector<double> temp;
		for (int j = 0; j < rowNum; ++j)
		{
			temp.push_back(matrix[j][i]);	
		}
		matrixT.push_back(temp);
	} 		
}

void matrix_multiply(const vector<vector<double> >& matrixA, const vector<vector<double> >& matrixB,  vector<vector<double> >& matrixResult)
{
	if (matrixA.empty() || matrixB.empty())
		return;	
	int rowNumA = matrixA.size();
	int colNumA = matrixA[0].size();
    int rowNumB = matrixB.size();
	int colNumB = matrixB[0].size();
	if (rowNumA != colNumB || colNumA != rowNumB)
		return;	
	for (int i = 0; i < rowNumA; ++i)
	{
		vector<double> temp;
		temp.assign(colNumB, 0.0);
		matrixResult.push_back(temp);
	}
    for (int i = 0; i < rowNumA; ++i)
	{
		for (int j = 0; j < colNumB; ++j)
		{
			for (int k = 0; k < colNumA; ++k)
				matrixResult[i][j] += matrixA[i][k] * matrixB[k][j]; 	
		}
	}
}

void print_vector(const vector<double>& t)
{
	for (int i = 0; i < t.size(); i++)
	{
		cout << t[i] << ",";
	}
	cout << endl;
}

void print_matrix(const vector<vector<double> >& matrix)
{
	for (int i = 0; i < matrix.size(); i++)
	{
		print_vector(matrix[i]);
	}	
}

void loadFromFile(vector<pair<vector<double>, int> >&  sample, const char* filePath)
{
	ifstream in(filePath);
	string line;
    while (getline(in, line))
	{
		size_t pos = line.find_first_of('\t');
		splitstring str(line.substr(0, pos));			
		vector<double> x;
		vector<string> flds = str.split(' ');
		int i = 0;
		x.push_back(1.0);
		for (int i = 0; i < flds.size(); i++) 
		{
			x.push_back(atof(flds[i].c_str()));
		}
		int y = atoi(line.substr(pos + 1).c_str());
		sample.push_back(make_pair(x, y));
	}
}

void toMatrixXf(const vector<vector<double> >& matrix, int rowNum, int colNum, MatrixXf& m)
{
	m =  MatrixXf::Zero(rowNum, colNum);
	for (int i = 0; i < rowNum; ++i)
		for (int j = 0; j< colNum; ++j)
			m(i, j) = matrix[i][j];
}

void ridge_regression(vector<pair<vector<double>, int> >& trainData, double lambda, vector<double>& weight) 
{
	vector<vector<double> > fMatrix;
	vector<vector<double> > yMatrix;
	for (int i = 0; i < trainData.size(); ++i)
	{
		vector<double>& temp = trainData[i].first;
		vector<double> yTemp;
		yTemp.push_back(trainData[i].second);
		fMatrix.push_back(temp);
		yMatrix.push_back(yTemp);
	}

	if (fMatrix.empty() || yMatrix.empty())
		return;

    int fRowNum = fMatrix.size();
    int fColNum = fMatrix[0].size();
	MatrixXf fEigenMatrix;
	toMatrixXf(fMatrix, fRowNum, fColNum, fEigenMatrix);

    MatrixXf tempEigenMatrix;
	tempEigenMatrix = fEigenMatrix.transpose() * fEigenMatrix + lambda * MatrixXf::Identity(fColNum, fColNum);

	int yRowNum = yMatrix.size();
    int yColNum = yMatrix[0].size();
	MatrixXf yEigenMatrix;
	toMatrixXf(yMatrix, yRowNum, yColNum, yEigenMatrix);

	MatrixXf weightEigenMatrix;
	weightEigenMatrix = tempEigenMatrix.inverse() * fEigenMatrix.transpose() * yEigenMatrix;

    for (int i = 0; i < fColNum; ++i)
	{
		cout << "weight" << weightEigenMatrix(i,0) << endl;
		weight.push_back(weightEigenMatrix(i, 0));
	}
}

double predict(vector<pair<vector<double>, int> >& testData, vector<double>& weight)
{
	double error = 0.0;
	if (testData.empty())
		return -1;
	int errorCase = 0;
 	for (int i = 0; i < testData.size(); ++i)
	{
		vector<double>& fVec = testData[i].first;
		double predictValue = vector_dot_product(fVec, weight);
		if (predictValue * testData[i].second < 0)
		{
			errorCase++;
		}	
	}	
	return (double)errorCase / testData.size();
}

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		cerr << "./rr TRAIN_DATA_FILE TEST_DATA_FILE lambda" << endl;
		return -1;
	}
	vector<pair<vector<double>, int> > trainData;
	vector<pair<vector<double>, int> > testData;
    loadFromFile(trainData, argv[1]);		
    loadFromFile(testData, argv[2]);		
    double lambda = atof(argv[3]);		
    cout << lambda << endl;
    vector<double> weight;
    ridge_regression(trainData, lambda, weight); 
    cout << "E_in : " << predict(trainData, weight) << endl;
    cout << "E_out : " << predict(testData, weight) << endl;
	return 0;
}
