#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<set>
#include<utility>
#include<cstdlib>
#include<ctime>

using namespace std;

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

//x vector and weight vector should have the same length
void pla(vector<pair<vector<double>, int> >& sample, vector<double>& weight)
{
	int round = 0;
	int count = 0;
	for (int i = 0; i < sample.size(); i++)
	{
		if(sample[i].second * vector_dot_product(sample[i].first, weight) <= 0)
		{
			for (int j = 0; j < weight.size(); j++)
				weight[j] = weight[j] + 0.5 * sample[i].second * sample[i].first[j];
			i = 0;
		}
	}	
	
}

void getRandomSeq(vector<int>& randomIndex, int range)
{
	set<int> checkSet;
	srand((unsigned)time(0));
	randomIndex.clear();
	while (true)
	{
		int number = rand()%range;
		if (checkSet.find(number) == checkSet.end())
		{
			randomIndex.push_back(number);
		}
		checkSet.insert(number);	
		if (randomIndex.size() == range)
			break;
	}
}

//x vector and weight vector should have the same length
int pla_random(vector<pair<vector<double>, int> >& sample, vector<double>& weight)
{
	//random visit	
	vector<int> randomIndex;
	getRandomSeq(randomIndex, sample.size());	
	int iter = 0;
	int count = 0;
	for (int i = 0; i < randomIndex.size(); ++i)
	{
		if(sample[randomIndex[i]].second * vector_dot_product(sample[randomIndex[i]].first, weight) <= 0)
		{
			for (int j = 0; j < weight.size(); j++)
				weight[j] = weight[j] + sample[randomIndex[i]].second * sample[randomIndex[i]].first[j];
			iter++;
			if (iter >= 50)
				break;
			i = 0;
		}
	}
	return iter;
}

double predict(vector<pair<vector<double>, int> >& testData, vector<double>& weight)
{
	if (testData.empty())
		return 0.0;
	int error = 0;
	for (int i = 0; i < testData.size(); ++i)
	{
		if (testData[i].second * vector_dot_product(testData[i].first, weight) <= 0)	
			error++;
	}
	return (double)error / testData.size();
}

double pla_pocket(vector<pair<vector<double>, int> >& sample, vector<double>& weight,  int maxIter)
{
	double bestErrorRate = 1;
	vector<double> finalWeight;
	vector<int> randomIndex;
	getRandomSeq(randomIndex, sample.size());	
	int iter = 0;
	int count = 0;
	for (int i = 0; i < randomIndex.size(); ++i)
	{
		if(sample[randomIndex[i]].second * vector_dot_product(sample[randomIndex[i]].first, weight) <= 0)
		{
			for (int j = 0; j < weight.size(); j++)
				weight[j] = weight[j] + sample[randomIndex[i]].second * sample[randomIndex[i]].first[j];
			double err = predict(sample, weight);
			if (err < bestErrorRate)
			{
				bestErrorRate = err;
				finalWeight = weight;
				iter++;
				getRandomSeq(randomIndex, sample.size());	
			}
			if (iter >= maxIter)	
				break;
		}
	}
	return bestErrorRate;
}


void print_vector(vector<double>& t)
{
	for (int i = 0; i < t.size(); i++)
	{
		cout << t[i] << ",";
	}
	cout << endl;
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


int main(int argc, char** argv)
{
	if (argc != 3)
		cerr << "./pla TRAIN_DATA_FILE TEST_DATA_FILE" << endl;
	vector<pair<vector<double>, int> > trainData;
	vector<pair<vector<double>, int> > testData;
        loadFromFile(trainData, argv[1]);		
        loadFromFile(testData, argv[2]);		
	//pla(sample, weight);
	/*
	int i = 0;
	int count = 0;
	while (i++ <= 2000)
	{
		weight.clear();
		weight.assign(5, 0.0);
		count += pla_random(sample, weight);
		sleep(1);
	}
	cout << "avg update " << count / (i - 1) << " times" << endl;
	*/
	vector<double> weight;
	double sumErrRate = 0.0;
	int times = 2000;
	for (int i = 0; i < times; ++i)
	{
		weight.clear();
		weight.assign(5, 0.0);
		pla_pocket(trainData, weight, 100);
		//pla_random(trainData,weight);
		double errRate = predict(testData, weight); 
		sumErrRate += errRate;
		cout << "round " << i << ": " << errRate << endl;
		sleep(1);
	}
	cout << "avg err is " << sumErrRate/times << endl;
	return 0;
}
