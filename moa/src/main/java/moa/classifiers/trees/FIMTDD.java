/*
 *    FIMTDD.java
 *    Copyright (C) 2015 Jožef Stefan Institute, Ljubljana, Slovenia
 *    @author Aljaž Osojnik <aljaz.osojnik@ijs.si>
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 *    
 */

package moa.classifiers.trees;

import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Instance;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import moa.options.ClassOption;
import moa.classifiers.Regressor;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.FIMTDDNumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.SizeOf;

/*
 * Implementation of FIMTDD, regression and model trees for data streams.
 */

public class FIMTDD extends AbstractClassifier implements Regressor {

	private static final long serialVersionUID = 1L;

	protected FIMTDDNode treeRoot;

	protected int leafNodeCount = 0;
	protected int splitNodeCount = 0;

	protected double examplesSeen = 0.0;
	protected double sumOfValues = 0.0;
	protected double sumOfSquares = 0.0;

	protected DoubleVector sumOfAttrValues = new DoubleVector();
	protected DoubleVector sumOfAttrSquares = new DoubleVector();

	public int maxID = 0;

	//region ================ OPTIONS ================

	public ClassOption splitCriterionOption = new ClassOption(
			"splitCriterion", 's', "Split criterion to use.",
			SplitCriterion.class, "moa.classifiers.core.splitcriteria.VarianceReductionSplitCriterion");

	public IntOption gracePeriodOption = new IntOption(
			"gracePeriod", 'g', "Number of instances a leaf should observe between split attempts.",
			200, 0, Integer.MAX_VALUE);

	public FloatOption splitConfidenceOption = new FloatOption(
			"splitConfidence", 'c', "Allowed error in split decision, values close to 0 will take long to decide.",
			0.0000001, 0.0, 1.0);

	public FloatOption tieThresholdOption = new FloatOption(
			"tieThreshold", 't', "Threshold below which a split will be forced to break ties.",
			0.05, 0.0, 1.0);

	public FloatOption PageHinckleyAlphaOption = new FloatOption(
			"PageHinckleyAlpha", 'a', "Alpha value to use in the Page Hinckley change detection tests.",
			0.005, 0.0, 1.0);

	public IntOption PageHinckleyThresholdOption = new IntOption(
			"PageHinckleyThreshold", 'h', "Threshold value used in the Page Hinckley change detection tests.",
			50, 0, Integer.MAX_VALUE);

	public FloatOption alternateTreeFadingFactorOption = new FloatOption(
			"alternateTreeFadingFactor", 'f', "Fading factor used to decide if an alternate tree should replace an original.",
			0.995, 0.0, 1.0);

	public IntOption alternateTreeTMinOption = new IntOption(
			"alternateTreeTMin", 'y', "Tmin value used to decide if an alternate tree should replace an original.",
			150, 0, Integer.MAX_VALUE);

	public IntOption alternateTreeTimeOption = new IntOption(
			"alternateTreeTime", 'u', "The number of instances used to decide if an alternate tree should be discarded.",
			1500, 0, Integer.MAX_VALUE);

	public FlagOption regressionTreeOption = new FlagOption(
			"regressionTree", 'e', "Build a regression tree instead of a model tree.");

	public FloatOption learningRatioOption = new FloatOption(
			"learningRatio", 'l', "Learning ratio to used for training the Perceptrons in the leaves.",
			0.02, 0, 1.00);

	public FloatOption learningRateDecayFactorOption = new FloatOption(
			"learningRatioDecayFactor", 'd', "Learning rate decay factor (not used when learning rate is constant).",
			0.001, 0, 1.00);

	public FlagOption learningRatioConstOption = new FlagOption(
			"learningRatioConst", 'p', "Keep learning rate constant instead of decaying.");

	//endregion ================ OPTIONS ================

	//region ================ METHODS ================

	// region --- Regressor methods
	
	public String getPurposeString() {
		return "Implementation of the FIMT-DD tree as described by Ikonomovska et al.";
	}

	public void resetLearningImpl() {
		this.treeRoot = null;
		this.leafNodeCount = 0;
		this.splitNodeCount = 0;
		this.maxID = 0;
		this.examplesSeen = 0;
		this.sumOfValues = 0.0;
		this.sumOfSquares = 0.0;

		this.sumOfAttrValues = new DoubleVector();
		this.sumOfAttrSquares = new DoubleVector();
	}

	public boolean isRandomizable() {
		return true;
	}

	public void getModelDescription(StringBuilder out, int indent) {
		if (treeRoot != null) treeRoot.describeSubtree(out, indent);
	}

	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{ 
				new Measurement("tree size (leaves)", this.leafNodeCount)
		};
	}

	public int calcByteSize() {
		return (int) SizeOf.fullSizeOf(this);
	}

	public double[] getVotesForInstance(Instance inst) {
		if (treeRoot == null) {
			return new double[] {0};
		}

		double prediction = treeRoot.getPrediction(inst);

		return new double[] {prediction};
	}

	public double normalizeTargetValue(double value) {
		if (examplesSeen > 1) {
			double sd = Math.sqrt((sumOfSquares - ((sumOfValues * sumOfValues)/examplesSeen))/examplesSeen);
			double average = sumOfValues / examplesSeen;
			if (sd > 0 && examplesSeen > 1)
				return (value - average) / (3 * sd);
			else
				return 0.0;
		}
		return 0.0;
	}

	public double getNormalizedError(Instance inst, double prediction) {
		double normalPrediction = normalizeTargetValue(prediction);
		double normalValue = normalizeTargetValue(inst.classValue());
		return Math.abs(normalValue - normalPrediction);
	}


	/**
	 * Method for updating (training) the model using a new instance
	 */
	public void trainOnInstanceImpl(Instance inst) {
		checkRoot();

		examplesSeen += inst.weight();
		sumOfValues += inst.weight() * inst.classValue();
		sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();

		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			int aIndex = modelAttIndexToInstanceAttIndex(i, inst);
			sumOfAttrValues.addToValue(i, inst.weight() * inst.value(aIndex));
			sumOfAttrSquares.addToValue(i, inst.weight() * inst.value(aIndex) * inst.value(aIndex));
		}

		double prediction = treeRoot.getPrediction(inst);
		processInstance(inst, treeRoot, prediction, getNormalizedError(inst, prediction), true, false);
	}

	public void processInstance(Instance inst, FIMTDDNode node, double prediction, double normalError, boolean growthAllowed, boolean inAlternate) {
		FIMTDDNode currentNode = node;
		while (true) {
			if (currentNode instanceof FFMTDDLeafNode) {
				((FFMTDDLeafNode) currentNode).learnFromInstance(inst, growthAllowed);
				break;
			} else {
				currentNode.examplesSeen += inst.weight();
				currentNode.sumOfAbsErrors += inst.weight() * normalError;
				FIMTDDSplitNode iNode = (FIMTDDSplitNode) currentNode;
				if (!inAlternate && iNode.alternateTree != null) {
					boolean altTree = true;
					double lossO = Math.pow(inst.classValue() - prediction, 2);
					double lossA = Math.pow(inst.classValue() - iNode.alternateTree.getPrediction(inst), 2);
					
					// Loop for compatibility with bagging methods
					for (int i = 0; i < inst.weight(); i++) {
						iNode.lossFadedSumOriginal = lossO + alternateTreeFadingFactorOption.getValue() * iNode.lossFadedSumOriginal;
						iNode.lossFadedSumAlternate = lossA + alternateTreeFadingFactorOption.getValue() * iNode.lossFadedSumAlternate;
						iNode.lossExamplesSeen++;
						
						double Qi = Math.log((iNode.lossFadedSumOriginal) / (iNode.lossFadedSumAlternate));
						iNode.lossSumQi += Qi;
						iNode.lossNumQiTests += 1;
					}
					double Qi = Math.log((iNode.lossFadedSumOriginal) / (iNode.lossFadedSumAlternate));
					double previousQiAverage = iNode.lossSumQi / iNode.lossNumQiTests;
					double QiAverage = iNode.lossSumQi / iNode.lossNumQiTests;
					
					if (iNode.lossExamplesSeen - iNode.previousWeight >= alternateTreeTMinOption.getValue()) {
						iNode.previousWeight = iNode.lossExamplesSeen;
						if (Qi > 0) {
							// Switch the subtrees
							FIMTDDNode parent = currentNode.getParent();

							if (parent != null) {
								FIMTDDNode replacementTree = iNode.alternateTree;
								parent.setChild(parent.getChildIndex(currentNode), replacementTree);
								if (growthAllowed) replacementTree.restartChangeDetection();
							} else {
								treeRoot = iNode.alternateTree;
								treeRoot.restartChangeDetection();
							}

							currentNode = iNode.alternateTree;
							currentNode.originalNode = null;
							altTree = false;
						} else if (
								(QiAverage < previousQiAverage && iNode.lossExamplesSeen >= (10 * this.gracePeriodOption.getValue()))
								|| iNode.lossExamplesSeen >= alternateTreeTimeOption.getValue()
								) {
							// Remove the alternate tree
							iNode.alternateTree = null;
							if (growthAllowed) iNode.restartChangeDetection();
							altTree = false;
						}
					}

					if (altTree) {
						growthAllowed = false;
						processInstance(inst, iNode.alternateTree, prediction, normalError, true, true);
					}
				}

				if (iNode.changeDetection && !inAlternate) {
					if (iNode.PageHinckleyTest(normalError - iNode.sumOfAbsErrors / iNode.examplesSeen - PageHinckleyAlphaOption.getValue(), PageHinckleyThresholdOption.getValue())) {
						iNode.initializeAlternateTree();
					}
				}
				if (currentNode instanceof FIMTDDSplitNode) {
					currentNode = ((FIMTDDSplitNode) currentNode).descendOneStep(inst);
				} 
			}
		}
	}

	// endregion --- Regressor methods
	
	// region --- Object instatiation methods

	protected FIMTDDNumericAttributeClassObserver newNumericClassObserver() {
		return new FIMTDDNumericAttributeClassObserver();
	}

	protected FIMTDDSplitNode newSplitNode(InstanceConditionalTest splitTest) {
		maxID++;
		return new FIMTDDSplitNode(splitTest, this);
	}

	protected FFMTDDLeafNode newLeafNode() {
		maxID++;
		return new FFMTDDLeafNode(this) {
			@Override
			protected int modelAttIndexToInstanceAttIndex(int i, Instance instance) {
				return AbstractClassifier.modelAttIndexToInstanceAttIndex(i,instance);
			}
		};
	}

	protected FIMTDDPerceptron newLeafModel() {
		FIMTDDPerceptron fimtddPerceptron = new FIMTDDPerceptron(this) {
			@Override
			public double getExamplesSeen() {
				return examplesSeen;
			}
		};
		fimtddPerceptron.learningRatioOption = this.learningRatioOption;
		return fimtddPerceptron;
	}

	//endregion --- Object instatiation methods
	
	// region --- Processing methods
	
	protected void checkRoot() {
		if (treeRoot == null) {
			treeRoot = newLeafNode();
			leafNodeCount = 1;
		}
	}

	public static double computeHoeffdingBound(double range, double confidence, double n) {
		return Math.sqrt(( (range * range) * Math.log(1 / confidence)) / (2.0 * n));
	}

	public boolean buildingModelTree() {
		return !regressionTreeOption.isSet();
	}

	protected void attemptToSplit(FFMTDDLeafNode node, FIMTDDNode parent, int parentIndex) {
		// Set the split criterion to use to the SDR split criterion as described by Ikonomovska et al. 
		SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);

		// Using this criterion, find the best split per attribute and rank the results
		AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion);
		Arrays.sort(bestSplitSuggestions);

		// Declare a variable to determine if any of the splits should be performed
		boolean shouldSplit = false;

		// If only one split was returned, use it
		if (bestSplitSuggestions.length < 2) {
			shouldSplit = bestSplitSuggestions.length > 0;
		} else { // Otherwise, consider which of the splits proposed may be worth trying

			// Determine the Hoeffding bound value, used to select how many instances should be used to make a test decision
			// to feel reasonably confident that the test chosen by this sample is the same as what would be chosen using infinite examples
			double hoeffdingBound = computeHoeffdingBound(1, this.splitConfidenceOption.getValue(), node.examplesSeen);
			// Determine the top two ranked splitting suggestions
			AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
			AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

			// If the upper bound of the sample mean for the ratio of SDR(best suggestion) to SDR(second best suggestion),
			// as determined using the Hoeffding bound, is less than 1, then the true mean is also less than 1, and thus at this
			// particular moment of observation the bestSuggestion is indeed the best split option with confidence 1-delta, and
			// splitting should occur.
			// Alternatively, if two or more splits are very similar or identical in terms of their splits, then a threshold limit
			// (default 0.05) is applied to the Hoeffding bound; if the Hoeffding bound is smaller than this limit then the two
			// competing attributes are equally good, and the split will be made on the one with the higher SDR value.
			if ((secondBestSuggestion.merit / bestSuggestion.merit < 1 - hoeffdingBound) || (hoeffdingBound < this.tieThresholdOption.getValue())) {
				shouldSplit = true;
			}
			// If the splitting criterion was not met, initiate pruning of the E-BST structures in each attribute observer
			else {
				for (int i = 0; i < node.attributeObservers.size(); i++) {
					FIMTDDNumericAttributeClassObserver obs = node.attributeObservers.get(i);
					if (obs != null) {
						obs.removeBadSplits(splitCriterion, secondBestSuggestion.merit / bestSuggestion.merit, bestSuggestion.merit, hoeffdingBound);    
					}
				}
			}
		}

		// If the splitting criterion were met, split the current node using the chosen attribute test, and
		// make two new branches leading to (empty) leaves
		if (shouldSplit) {
			AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];

			FIMTDDSplitNode newSplit = newSplitNode(splitDecision.splitTest);
			newSplit.copyStatistics(node);
			newSplit.changeDetection = node.changeDetection;
			newSplit.ID = node.ID;
			leafNodeCount--;
			for (int i = 0; i < splitDecision.numSplits(); i++) {
				FFMTDDLeafNode newChild = newLeafNode();
				if (buildingModelTree()) {
					// Copy the splitting node's perceptron to it's children
					newChild.learningModel = new FIMTDDPerceptron((FIMTDDPerceptron) node.learningModel);
					
				}
				newChild.changeDetection = node.changeDetection;
				newChild.setParent(newSplit);
				newSplit.setChild(i, newChild);
				leafNodeCount++;
			}
			if (parent == null && node.originalNode == null) {
				treeRoot = newSplit;
			} else if (parent == null && node.originalNode != null) {
				node.originalNode.alternateTree = newSplit;
			} else {
				((FIMTDDSplitNode) parent).setChild(parentIndex, newSplit);
				newSplit.setParent(parent);
			}
			
			splitNodeCount++;
		}
	}
	

	//endregion --- Processing methods
	
	//endregion ================ METHODS ================
}

