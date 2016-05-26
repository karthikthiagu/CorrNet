__author__ = 'Sarath'

import numpy
import os
import sys

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)


def get_mat(fname):

	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = numpy.asarray(mat,dtype="float32")
	return mat


def get_mat1(folder,mat1,mat2):
	file = open(folder+"ip.txt","w")
	flag = 0

	smat1 = numpy.zeros((1000,len(mat1[0])))
	smat2 = numpy.zeros((1000,len(mat2[0])))
	i = 0
	while(i!=len(mat1)):
		flag = 1
		smat1[i%1000] = mat1[i]
		smat2[i%1000] = mat2[i]
		i+=1
		if(i%1000==0):
			numpy.save(folder+str(i/1000)+"_left",smat1)
			numpy.save(folder+str(i/1000)+"_right",smat2)
			file.write("xy,dense,"+folder+str(i/1000)+",1000\n")
			smat1 = numpy.zeros((1000,len(mat1[0])))
			smat2 = numpy.zeros((1000,len(mat2[0])))
			flag = 0

	if(flag!=0):
		numpy.save(folder+str((i/1000) +1)+"_left",smat1)
		numpy.save(folder+str((i/1000) +1)+"_right",smat2)
		file.write("xy,dense,"+folder+str((i/1000) +1)+","+str(i%1000)+"\n")
	file.close()

def converter(folder):
    
	create_folder(folder+"matpic/")
	create_folder(folder+"matpic1/")
    for stage in ['train', 'valid', 'test']:
        create_folder(folder + 'matpic/%s' % stage)
        create_folder(folder + 'matpic1/%s' % stage)

    	mat1 = get_mat(folder+"%s_view1_features.txt" % stage)
	    numpy.save(folder+"matpic/%s/view1" % stage, mat1)
	    mat2 = get_mat(folder+"%s_view2_features.txt" % stage)
	    numpy.save(folder+"matpic/%s/view2" % stage, mat2)
        get_mat1(folder+"matpic1/%s/" % stage, mat1, mat2)
	    numpy.save(folder+"matpic/%s/labels" % stage, get_mat(folder+"%s_view1_labels.txt" % stage))

converter(sys.argv[1])
