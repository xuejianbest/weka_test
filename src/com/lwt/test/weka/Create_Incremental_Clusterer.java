package com.lwt.test.weka;

import java.io.File;

import weka.clusterers.Cobweb;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * 创建增量聚类器——
 * 实现了weka.clusterers.UpdateableClusterer接口的聚类器可以被增量地训练，
 *
 * 先用一个空的（或非空）数据集训练；
 * 然后再调用updateClusterer(instance)一个个的训练；
 * 最后调用updateFinished()完成最后计算。
 */
public class Create_Incremental_Clusterer extends BaseTest{

	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(data_path+"houses.arff"));
		
		//创建一个只有头的空实例结构
		Instances dataset = loader.getStructure();
		
		Cobweb cluster_cw_u = new Cobweb();
		cluster_cw_u.buildClusterer(dataset); 
		
		Instance current; 
		while ((current = loader.getNextInstance(dataset)) != null)
			cluster_cw_u.updateClusterer(current); 
		
		cluster_cw_u.updateFinished(); 
		System.out.println(cluster_cw_u);
	}
}
