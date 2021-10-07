#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
#include<string>
#include<sstream>
#include<math.h>
using namespace std;


struct Point3d
{
	double x[3];
	Point3d()
	{
		for(int i=0;i<3;i++)
			x[i] = 0;
	}
	inline Point3d operator-(const Point3d&tar)
	{
		Point3d res;
		for(int i=0;i<3;i++)
			res.x[i] = x[i] - tar.x[i];
		return res;
	}
	inline Point3d operator+(const Point3d&tar)
	{
		Point3d res;
		for(int i=0;i<3;i++)
			res.x[i] = x[i] + tar.x[i];
		return res;
	}
	inline Point3d operator*(const double&tar)
	{
		Point3d res;
		for(int i=0;i<3;i++)
			res.x[i] = x[i] * tar;
		return res;
	}
	inline Point3d operator/(const double&tar)
	{
		Point3d res;
		for(int i=0;i<3;i++)
			res.x[i] = x[i] / tar;
		return res;
	}
	inline double len()
	{
		double res = 0;
		for(int i=0;i<3;i++)
			res += x[i]*x[i];
		return sqrt(res);
	}
	inline Point3d unit()
	{
		Point3d res;
		double tmp_len = len();
		for(int i=0;i<3;i++)
			res.x[i] = x[i]/tmp_len;
		return res;
	}
	inline double cross(const Point3d&tar)
	{
		double res = 0;
		for(int i=0;i<3;i++)
			res += x[i]*tar.x[i];
		return res;
	}

	void print()
	{
		for(int i=0;i<3;i++)
			cout<<x[i]<<" ";
		cout<<endl;
	}
};

struct Face3d
{
	int x[3];
	Face3d(int x0=0, int x1=0, int x2=0)
	{
		x[0] = x0;x[1] = x1; x[2] = x2;
	}
};



const int EYE_POINTS_NUM = 34;
const int EYE_PRECIOUS = 10;

class MeshList
{
public:
	vector<Point3d> vertices;
	vector<Face3d> faces;
	vector<int> eye_rank[2];
	MeshList(char* eye_file)
	{
		ifstream fin(eye_file);
		for(int i=0;i<2;i++)
		{
			for(int j=0;j<EYE_POINTS_NUM;j++)
			{
				int tmp_r;
				fin>>tmp_r;
				eye_rank[i].push_back(tmp_r);
			}
		}
	}
	void readObj(char* filename)
	{
		ifstream fin(filename);
		string line_str;
		while(getline(fin, line_str))
		{
			stringstream ss(line_str);
			char line_type;
			ss>>line_type;
			if(line_type == 'v')
			{
				Point3d tmp_point3d;
				ss>>tmp_point3d.x[0]>>tmp_point3d.x[1]>>tmp_point3d.x[2];
				vertices.push_back(tmp_point3d);
			}
			if(line_type == 'f')
			{
				Face3d tmp_face3d;
				ss>>tmp_face3d.x[0]>>tmp_face3d.x[1]>>tmp_face3d.x[2];
				faces.push_back(tmp_face3d);
			}
		}
	}

	void add_eye()
	{
		for(int i=0;i<2;i++)
		{
			Point3d mean;
			for(int j=0;j<EYE_POINTS_NUM;j++)
			{
				mean = mean + vertices[eye_rank[i][j]];
			}
			mean = mean/EYE_POINTS_NUM;
			for(int k=1;k<EYE_PRECIOUS;k++)
			{
				for(int j=0;j<EYE_POINTS_NUM;j++)
				{
					Point3d tmp_p3d;
					tmp_p3d = mean*k + vertices[eye_rank[i][j]]*(EYE_PRECIOUS-k);
					tmp_p3d = tmp_p3d/EYE_PRECIOUS;
					eye_rank[i].push_back(vertices.size());
					vertices.push_back(tmp_p3d);
				}
			}
			for(int k=0;k<(EYE_PRECIOUS-1);k++)
			{
				for(int j=0;j<EYE_POINTS_NUM;j++)
				{
					faces.push_back(Face3d(eye_rank[i][EYE_POINTS_NUM*k+j]+1, eye_rank[i][EYE_POINTS_NUM*k+(j+1)%EYE_POINTS_NUM]+1, eye_rank[i][EYE_POINTS_NUM*(k+1)+(j+1)%EYE_POINTS_NUM]+1));
				}
			}
			for(int k=1;k<EYE_PRECIOUS;k++)
			{
				for(int j=0;j<EYE_POINTS_NUM;j++)
				{
					faces.push_back(Face3d(eye_rank[i][EYE_POINTS_NUM*k+(j+1)%EYE_POINTS_NUM]+1, eye_rank[i][EYE_POINTS_NUM*k+j]+1, eye_rank[i][EYE_POINTS_NUM*(k-1)+(j)%EYE_POINTS_NUM]+1));
				}
			}
			vertices.push_back(mean);
			for(int j=0;j<EYE_POINTS_NUM;j++)
			{
				faces.push_back(Face3d(eye_rank[i][EYE_POINTS_NUM*(EYE_PRECIOUS-1)+j]+1, eye_rank[i][EYE_POINTS_NUM*(EYE_PRECIOUS-1)+(j+1)%EYE_POINTS_NUM]+1, vertices.size()));
			}
		}
	}

	void saveObj(char* filename)
	{
		ofstream fout(filename);
		fout<<"# "<<vertices.size()<<" vertices, "<<faces.size()<<" faces"<<endl;
		for(int i=0;i<vertices.size();i++)
		{
			fout<<setprecision(9)<<"v "<<vertices[i].x[0]<<" "<<vertices[i].x[1]<<" "<<vertices[i].x[2]<<endl;
		}
		for(int i=0;i<faces.size();i++)
		{
			fout<<"f "<<faces[i].x[0]<<" "<<faces[i].x[1]<<" "<<faces[i].x[2]<<endl;
		}
	}
};