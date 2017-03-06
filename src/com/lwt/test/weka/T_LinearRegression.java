package com.lwt.test.weka;

import java.io.File;

import weka.classifiers.functions.LinearRegression;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class T_LinearRegression extends BaseTest {
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(data_path+"houses.arff"));
		Instances dataset = loader.getDataSet();
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		LinearRegression model = new LinearRegression();
		model.buildClassifier(dataset);
		
		DenseInstance instance = new DenseInstance(dataset.numAttributes());
		double[] values = {3198, 9669, 5, 1, 1};
		for(int i=0; i<values.length; i++){
			instance.setValue(i, values[i]);
		}
		
		double price = model.classifyInstance(instance);
		System.out.println(price);
	}
}