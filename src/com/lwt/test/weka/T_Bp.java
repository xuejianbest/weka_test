package com.lwt.test.weka;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class T_Bp extends BaseTest {
	public static void main(String[] args) throws IOException {
		// 便于测试，用数组保存一些数据，从数据库中取数据是同理的
		// 二维数组第一列表示当月的实际数据，第二列是上个月的数据，用于辅助对当月数据的预测的
		// 二维数组的数据用于测试集数据，为了展示两种weka载入数据的方法，将训练集数据从arff文件中读取
		double[][] a = { { -0.93, -0.995 }, { -0.93, -0.93 }, { -0.93, -0.93 },
				{ -0.95, -0.93 }, { -0.93, -0.95 }, { -0.95, -0.93 },
				{ -0.93, -0.95 }, { -0.93, -0.93 }, { -0.95, -0.93 },
				{ -0.9, -0.95 }, { -0.92, -0.9 }, { -0.575, -0.92 },
				{ -0.23, -0.575 } };

		// 读入训练集数据
		File inputFile = new File(data_path + "bp_train.arff");// 该文件见源代码最后的分享链接，可以下载后将路径替换掉
		ArffLoader atf = new ArffLoader();
		try {
			atf.setFile(inputFile);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		Instances instancesTrain = atf.getDataSet();
		instancesTrain.setClassIndex(0);// 设置训练数据集的类属性，即对哪个数据列进行预测（属性的下标从0开始）

		// 读入测试集数据
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();

		Attribute ratio = new Attribute("CUR", 1);// 创建属性，参数为属性名称和属性号，但属性号并不影响FastVector中属性的顺序
		Attribute preratio = new Attribute("PRE", 2);

		attrs.add(ratio);// 向FastVector中添加属性，属性在FastVector中的顺序由添加的先后顺序确定。
		attrs.add(preratio);

		Instances instancesTest = new Instances("bp", attrs, attrs.size());// 创建实例集，即数据集，参数为名称，FastVector类型的属性集，以及属性集的大小（即数据集的列数）

		instancesTest.setClass(ratio);// 设置数据集的类属性，即对哪个数据列进行预测

		for (int k = 0; k < 13; k++) {
			Instance ins = new DenseInstance(attrs.size());// 创建实例，即一条数据
			ins.setDataset(instancesTest);// 设置该条数据对应的数据集，和数据集的属性进行对应
			ins.setValue(ratio, a[k][0]);// 设置数据每个属性的值
			ins.setValue(preratio, a[k][1]);
			instancesTest.add(ins);// 将该条数据添加到数据集中
		}

		MultilayerPerceptron m_classifier = new MultilayerPerceptron();// 创建算法实例，要使用其他的算法，只用把类换做相应的即可

		try {
			m_classifier.buildClassifier(instancesTrain); // 进行训练
		} catch (Exception e) {
			e.printStackTrace();
		}

		for (int i = 0; i < 13; i++) {// 测试分类结果
			// instancesTest.instance(i)获得的是用模型预测的结果值，instancesTest.instance(i).classValue()获得的是测试集类属性的值
			// 此处是把预测值和实际值同时输出，进行对比
			try {
				System.out.println(m_classifier.classifyInstance(instancesTest.instance(i)) + " " + instancesTest.instance(i).classValue());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

}
