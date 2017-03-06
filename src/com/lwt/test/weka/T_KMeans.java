package com.lwt.test.weka;

import java.io.File;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class T_KMeans extends BaseTest{
	/**
	 *  -init 0 = random, 1 = k-means++, 2 = canopy, 3 = farthest first.
	 *  -max-candidates 100 
	 *	-periodic-pruning 10000 
	 *	-min-density 2.0 
	 *	-t1 -1.25 
	 *	-t2 -1.0 
	 *	-A "weka.core.ManhattanDistance -R first-last" 
	 *	-I 500 
	 *  -num-slots 1 
	 */
	public static void main(String[] args) throws Exception{
		CSVLoader loader = new CSVLoader();
		loader.setFile(new File(data_path+"kmeansdata.csv"));
		Instances data = loader.getDataSet();
		
		SimpleKMeans model = new SimpleKMeans();
		//k=4, 最大迭代次数100, seed=12, 初始化方式k-means++
		String[] opt = {"-N", "4", "-I", "100", "-S", "12", "-init", "1"};  
		model.setOptions(opt);
		
		model.buildClusterer(data);
		System.out.println(model);
	}
	
}
