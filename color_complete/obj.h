#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
using namespace std;

int check_color(double x, double y, double z){
    if(x>0.95 && y>0.95 && z>0.95)
        return 0;
    return 1;
}

struct OBJ_color{
	//vertices
	vector<double> coord_x, coord_y, coord_z;
	vector<double> color_x, color_y, color_z;
	vector<int> unknown;// 1=unknown without color  2=unknow with color
	//faces and edges
	vector<int> face_a, face_b, face_c;
	vector<vector<int> > link_list;

	void read_obj(string filename){
		ifstream fin(filename);
		char type;
		while(fin>>type){
			if(type == 'v'){
				double x,y,z,cx,cy,cz;
				fin>>x>>y>>z>>cx>>cy>>cz;
				coord_x.push_back(x);
				coord_y.push_back(y);
				coord_z.push_back(z);
				color_x.push_back(cx);
				color_y.push_back(cy);
				color_z.push_back(cz);

				//unknown flag
				if(cx < 0)
					unknown.push_back(1);
                else if(check_color(cx, cy,cz) == 0)
                    unknown.push_back(1);
				else
					unknown.push_back(0);
				//link list initialization
				link_list.push_back(vector<int>());
			}
			if(type == 'f'){
				int a,b,c;
				fin>>a>>b>>c;
				face_a.push_back(a-1);
				face_b.push_back(b-1);
				face_c.push_back(c-1);
			}
		}
		cout<<coord_x.size()<<" "<<coord_y.size()<<" "<<coord_y.size()<<endl;
		cout<<color_x.size()<<" "<<color_y.size()<<" "<<color_z.size()<<endl;
		cout<<face_a.size()<<" "<<face_b.size()<<" "<<face_c.size()<<endl;
		fin.close();
	}

	void link_list_add(int rank, int tar){
		for(int i=0;i<link_list[rank].size();i++){
			if(link_list[rank][i] == tar)
				return;
		}
		link_list[rank].push_back(tar);
	}

	void build_link_list(){
		for(int i=0;i<face_a.size();i++){
			link_list_add(face_a[i], face_b[i]);
			link_list_add(face_a[i], face_c[i]);
			link_list_add(face_b[i], face_c[i]);
			link_list_add(face_b[i], face_a[i]);
			link_list_add(face_c[i], face_a[i]);
			link_list_add(face_c[i], face_b[i]);
		}
	}

	void color_spread_once(){
		int v_num = color_x.size();
		vector<double> tmp_color_x(v_num), tmp_color_y(v_num), tmp_color_z(v_num);
		vector<int> color_num(v_num);
		for(int i=0;i<v_num;i++){
			if(unknown[i] == 1)
				continue;
			for(int j=0;j<link_list[i].size();j++){
				int tar = link_list[i][j];
				if(unknown[tar] > 0){
					color_num[tar]++;
					tmp_color_x[tar] += color_x[i];
					tmp_color_y[tar] += color_y[i];
					tmp_color_z[tar] += color_z[i];
				}
				
			}
		}
		for(int i=0;i<v_num;i++){
			if(unknown[i] == 0)
				continue;
			if(color_num[i] == 0)
				continue;
			unknown[i] = 2;
			color_x[i] = tmp_color_x[i]/color_num[i];
			color_y[i] = tmp_color_y[i]/color_num[i];
			color_z[i] = tmp_color_z[i]/color_num[i];
		}
	}

	void color_spread(int times){
		for(int i=0;i<times;i++)
			color_spread_once();
	}

	void color_variance(){
		double v_num = color_x.size();
		double mx=0, my=0, mz=0;
		double vx=0, vy=0, vz=0;
		int mnum=0;
		for(int i=0;i<v_num;i++)
			if(unknown[i] == 0){
				mx += color_x[i];
				my += color_y[i];
				mz += color_z[i];
				mnum++;
			}
		mx = mx/mnum;
		my = my/mnum;
		mz = mz/mnum;
		for(int i=0;i<v_num;i++)
			if(unknown[i] == 0){
				vx += (color_x[i]-mx)*(color_x[i]-mx);
				vy += (color_y[i]-my)*(color_y[i]-my);
				vz += (color_z[i]-mz)*(color_z[i]-mz);
			}
		vx = sqrt(vx/mnum);
		vy = sqrt(vy/mnum);
		vz = sqrt(vz/mnum);
		cout<<vx<<" "<<vy<<" "<<vz<<endl;
		//
		default_random_engine e;
		uniform_real_distribution<double> n(-0.1,0.1);
		for(int i=0;i<v_num;i++)
			if(unknown[i] == 2){
				double ne = n(e);
				color_x[i] += vx*ne;
				color_y[i] += vy*ne;
				color_z[i] += vz*ne;
				if(color_x[i] < 0)
					color_x[i] = 0;
				if(color_y[i] < 0)
					color_y[i] = 0;
				if(color_z[i] < 0)
					color_z[i] = 0;
                if(color_x[i] > 1)
					color_x[i] = 1;
				if(color_y[i] > 1)
					color_y[i] = 1;
				if(color_z[i] > 1)
					color_z[i] = 1;
			}
	}


	void write_obj(string filename){
		ofstream fout(filename);
		for(int i=0;i<coord_x.size();i++){
			fout<<"v "<<coord_x[i]<<" "<<coord_y[i]<<" "<<coord_z[i]<<" "<<color_x[i]<<" "<<color_y[i]<<" "<<color_z[i]<<endl;
		}
		for(int i=0;i<face_a.size();i++){
			fout<<"f "<<face_a[i]+1<<" "<<face_b[i]+1<<" "<<face_c[i]+1<<endl;
		}
		fout.close();
	}
};