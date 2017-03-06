package com.lwt.test.weka;

import java.util.ArrayList;
import java.util.List;

import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.FPGrowth;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * FPGrowth的参数：
 * -N，是给出要输出多少条规则；
 * -T是指定选择哪个量进行排序，weka提供四种排序方法，0=confidence ，1=lift ， 2=leverage ， 3=Conviction。
 * -C是指你选定的那个排序参数的那个最小值， 要和-T一起设置，否则很容易挖掘不出规则
 * -M是是支持度的最小值，
 * -U是支持度的最大值。
 */
public class T_FPGrowth {
	
	public static void main(String[] args) throws Exception{
		String[][] items = {
				{"面包", "尿布", "啤酒"}, {"尿布", "啤酒"}, {"尿布", "啤酒"}, {"面包"}
		};
		
		int numAttributes = 0;
		for(String[] item : items){
			numAttributes = (int)Math.max(item.length, numAttributes);
		}
		
		List<String> attributeValues = new ArrayList<String>();
		attributeValues.add("1");
		
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		Attribute attr1 = new Attribute("面包", attributeValues);
		Attribute attr2 = new Attribute("尿布", attributeValues);
		Attribute attr3 = new Attribute("啤酒", attributeValues);
		attInfo.add(attr1);
		attInfo.add(attr2);
		attInfo.add(attr3);
		Instances dataset = new Instances("items", attInfo , items.length);
		
		for(int i=0; i<items.length; i++){
			Instance instance = new DenseInstance(numAttributes);
			instance.setDataset(dataset);
			for(int j=0; j<items[i].length; j++){
				instance.setValue(instance.dataset().attribute(items[i][j]), 0.0);
			}
			dataset.add(instance);
		}
		System.out.println(dataset);
		
		FPGrowth model = new FPGrowth();
		//支持度>=0。25,按置信度排序且置信度>0.8
		String[] options = Utils.splitOptions("-M 0.25 -T 0 -C 0.8");
		model.setOptions(options );
		model.buildAssociations(dataset);
		
		AssociationRules rules = model.getAssociationRules();
		
		System.out.println(model);
		System.out.println("=========");
		
		List<AssociationRule> rules_list = rules.getRules();
		AssociationRule rule = rules_list.get(0);
		System.out.println("..." + rule.toString());
	}

}
