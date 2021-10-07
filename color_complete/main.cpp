#include"obj.h"

#include<iostream>
#include<vector>
#include<random>
using namespace std;




int main(int argc, char *argv[])
{
	OBJ_color color;
	color.read_obj(argv[1]);
	color.build_link_list();
	color.color_spread(1000);
	color.color_variance();
	color.write_obj(argv[2]);
	return 0;
}