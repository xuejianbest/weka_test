package com.lwt.test.weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;

import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Instances;

public class T_TimeSeries extends BaseTest {
	public static void main(String[] args) throws Exception {
		Instances wine = new Instances(new BufferedReader(new FileReader(data_path+"wine.arff")));
		WekaForecaster forecaster = new WekaForecaster();

		// 要预测的属性名
		forecaster.setFieldsToForecast("Fortified,Dry-white");
		
		// 设置基础分类器，默认 SMOreg (SVM)
		forecaster.setBaseForecaster(new GaussianProcesses());

		forecaster.getTSLagMaker().setTimeStampField("Date"); // 时间戳属性名
		forecaster.getTSLagMaker().setMinLag(1);
		forecaster.getTSLagMaker().setMaxLag(12); //monthly data

		// add a month of the year indicator field
		forecaster.getTSLagMaker().setAddMonthOfYear(true);
		// add a quarter of the year indicator field
		forecaster.getTSLagMaker().setAddQuarterOfYear(true);

		forecaster.buildForecaster(wine, System.out);
		// 向模型填充数据
		forecaster.primeForecaster(wine);

		// 预测 5个未来值
		List<List<NumericPrediction>> forecast = forecaster.forecast(5);

		// 输出预测
		for (int i = 0; i < 5; i++) {
			List<NumericPrediction> predsAtStep = forecast.get(i);
			for (int j = 0; j < 2; j++) {
				NumericPrediction predForTarget = predsAtStep.get(j);
				System.out.print(predForTarget.predicted() + " ");
			}
			System.out.println();
		}
	}
}
