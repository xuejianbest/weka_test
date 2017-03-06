package com.lwt.test.weka;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * 验证模型
 * 交叉验证
 * 测试数据集验证
 */
public class T_Evaluating extends BaseTest{
	private static Classifier classifier = new J48();
	private static Instances trainset, testset;
	private static ArffLoader loader = new ArffLoader();
	
	public static void cross_validation() throws Exception{
		Evaluation eval = new Evaluation(trainset);
		eval.crossValidateModel(classifier, trainset, 10, new Random(1)); 
		System.out.println(eval.toSummaryString("Results\n=========\n", false)); 
	}
	
	public static void testset_validation() throws Exception{
		Evaluation eval = new Evaluation(testset);
		
		eval.evaluateModelOnce(classifier, testset.get(0));
		System.out.println(eval.toSummaryString("\nResults\n======\n", false)); 
	}
	
	public static void main(String[] args) throws Exception{
		loader.setFile(new File(data_path+"bmw-training.arff"));
		trainset = loader.getDataSet();
		trainset.setClassIndex(trainset.numAttributes()-1);
		
		classifier.buildClassifier(trainset);
		System.out.println(classifier);
		System.out.println("=========");
		
		cross_validation();
		loader.setFile(new File(data_path+"bmw-test.arff"));
		testset = loader.getDataSet();
		testset.setClassIndex(testset.numAttributes()-1);
		testset_validation();
	}

}
