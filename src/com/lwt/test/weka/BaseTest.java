package com.lwt.test.weka;

import java.io.File;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

public class BaseTest {
	protected static String data_path = new File("").getAbsolutePath()+File.separatorChar+"classes"+File.separatorChar;
	
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(data_path+"houses.arff"));
		Instances dataset = loader.getDataSet();
		
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		LinearRegression model = new LinearRegression();
		model.buildClassifier(dataset);
		
		SerializationHelper.write(data_path+"houses.model", model);
		System.out.println("model out done.");
	}

}
