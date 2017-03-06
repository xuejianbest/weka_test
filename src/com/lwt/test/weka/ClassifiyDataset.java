package com.lwt.test.weka;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

/**
 * 给一个未分类的样本分类
 * 并保存为arff文件
 */
public class ClassifiyDataset extends BaseTest {
	public static void main(String[] args) throws Exception {
		Instances unlabeled = new Instances(new BufferedReader(new FileReader(data_path+"unlabeled.arff"))); 
		Instances labeled = new Instances(unlabeled);
		labeled.setClassIndex(labeled.numAttributes()-1);
		
		Classifier classifier = (Classifier)SerializationHelper.read(data_path+"houses.model");
		
		for (int i=0; i<labeled.numInstances(); i++){
			Instance instance = labeled.instance(i);
		    double label = classifier.classifyInstance(instance);
		    instance.setClassValue(label); 
		}
		BufferedWriter writer = new BufferedWriter(new FileWriter(data_path+"labeled.arff"));
		writer.write(labeled.toString()); 
		writer.close();
		
		Instances output = new Instances(new BufferedReader(new FileReader(data_path+"labeled.arff"))); 
		System.out.println(output);
	}
}
