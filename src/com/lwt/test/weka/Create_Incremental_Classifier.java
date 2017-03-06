package com.lwt.test.weka;

import java.io.File;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * 创建增量分类器——
 * 实现了weka.classifiers.UpdateableClassifier接口的分类器可以被增量地训练，
 * 这样节省了内存，因为数据无需一次性地全部装入内存。
 *
 *先用一个空的数据集训练，然后在调用updateClassifier(instance)一个个的训练
 */
public class Create_Incremental_Classifier extends BaseTest {

	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(data_path+"breast-cancer.arff"));
		
		//创建一个只有头的空实例结构
		Instances dataset = loader.getStructure();
		/*
			@relation house
			
			@attribute houseSize numeric
			@attribute lotSize numeric
			@attribute bedrooms numeric
			@attribute granite numeric
			@attribute bathroom numeric
			@attribute sellingPrice numeric
			
			@data
		*/

		dataset.setClassIndex(dataset.numAttributes() - 1);
		NaiveBayesUpdateable model_nb_u = new NaiveBayesUpdateable();
		model_nb_u.buildClassifier(dataset);
		Instance current;
		while ((current = loader.getNextInstance(dataset)) != null)
			model_nb_u.updateClassifier(current);
		
		System.out.println(model_nb_u);
	}
}
