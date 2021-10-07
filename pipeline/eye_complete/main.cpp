#include<iostream>
#include"mesh.h"
using namespace std;

int main(int argc, char *argv[])
{
	//FaceMesh face("0.obj");
	//face.mark_border();
	MeshList mesh(argv[3]);
	mesh.readObj(argv[1]);
	mesh.add_eye();
	mesh.saveObj(argv[2]);
	return 0;
}