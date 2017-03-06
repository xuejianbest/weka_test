package com.lwt.test.weka;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class T_EM extends BaseTest{
  public static void main(String[] args) throws Exception {
    Instances data = DataSource.read(data_path + "iris.arff");
    data.setClassIndex(data.numAttributes() - 1);

    Remove filter = new Remove();
    filter.setAttributeIndices("" + (data.classIndex() + 1));
    filter.setInputFormat(data);
    Instances dataClusterer = Filter.useFilter(data, filter);

    EM clusterer = new EM();
    clusterer.buildClusterer(dataClusterer);

    ClusterEvaluation eval = new ClusterEvaluation();
    eval.setClusterer(clusterer);
    eval.evaluateClusterer(data);

    System.out.println(eval.clusterResultsToString());
  }
}