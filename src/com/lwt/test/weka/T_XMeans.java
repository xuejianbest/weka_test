package com.lwt.test.weka;

import java.io.File;

import weka.clusterers.XMeans;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class T_XMeans extends BaseTest{

	private ArffLoader loader;
	private Instances dataSet;
	private weka.clusterers.Clusterer cluster;
	private int numOfClusters;
	private File arffFile;
	private int sizeOfDataset;

	public T_XMeans(File arffFile) {
		this.arffFile = arffFile;
		doCluster();
	}

	private void doCluster() {
		loader = new ArffLoader();
		try {
			loader.setFile(arffFile);
			dataSet = loader.getDataSet();
			cluster = new XMeans();
			cluster.buildClusterer(dataSet);
			numOfClusters = cluster.numberOfClusters();
			sizeOfDataset = dataSet.numInstances();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public int clusterNewInstance(weka.core.Instance instance) {
		int indexOfCluster = -1;
		try {
			indexOfCluster = cluster.clusterInstance(instance);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return indexOfCluster;
	}

	public double[] frequencyOfCluster() {
		int[] sum = new int[this.numOfClusters];
		try {
			System.out.println("---------- will show the every instance's clusterIndex: ");
			for (int i = 0; i < this.sizeOfDataset; i++) {
				int clusterIndex = cluster.clusterInstance(dataSet.instance(i));
				sum[clusterIndex]++;
				System.out.println("instanceIndex: " + i + ", clusterIndex: " + clusterIndex + ",\t"
						+ dataSet.instance(i));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] fre = new double[sum.length];
		for (int i = 0; i < sum.length; i++) {
			fre[i] = (double) sum[i] / (double) this.sizeOfDataset;
		}
		return fre;
	}

	public static void main(String[] args) {
		File file = new File(data_path + "cpu.arff");
		T_XMeans wc = new T_XMeans(file);
		double[] fre = wc.frequencyOfCluster();
		for (int i = 0; i < fre.length; i++) {
			System.out.println("clusterIndex " + i + "'s freq: " + fre[i]);
		}

		double[] feature = { 125, 256, 6000, 256, 16, 128, 199 };
		Instance ins = new DenseInstance(7);
		for (int i = 0; i < ins.numAttributes(); i++) {
			ins.setValue(i, feature[i]);
		}
		System.out.println(ins + " in cluster: " + wc.clusterNewInstance(ins));
	}

}