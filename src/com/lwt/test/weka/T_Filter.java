package com.lwt.test.weka;

import java.io.File;

import weka.core.Instances; 
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter; 
import weka.filters.unsupervised.attribute.Remove; 
public class T_Filter extends BaseTest{

	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(data_path+"houses.arff"));
		Instances dataset = loader.getDataSet();
		System.out.println(dataset);
		System.out.println("========");
		String[] options = Utils.splitOptions("-R 1");
		Remove remove = new Remove();
		remove.setOptions(options);
		remove.setInputFormat(dataset);
		Instances newData = Filter.useFilter(dataset, remove); 
		System.out.println(newData);
	}

}
