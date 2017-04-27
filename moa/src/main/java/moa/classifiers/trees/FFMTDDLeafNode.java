package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.FIMTDDNumericAttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.StringUtils;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by Eduś on 2017-04-27.
 */
public abstract class FFMTDDLeafNode extends FIMTDDNode {

    private static final long serialVersionUID = 1L;

    // Perceptron model that carries out the actual learning in each node
    public FIMTDDPerceptron learningModel;

    protected AutoExpandVector<FIMTDDNumericAttributeClassObserver> attributeObservers = new AutoExpandVector<FIMTDDNumericAttributeClassObserver>();

    protected double examplesSeenAtLastSplitEvaluation = 0;

    /**
     * Create a new LeafNode
     */
    public FFMTDDLeafNode(FIMTDD tree) {
        super(tree);
        if (tree.buildingModelTree()) {
            learningModel = tree.newLeafModel();
        }
        examplesSeen = 0;
        sumOfValues = 0;
        sumOfSquares = 0;
        sumOfAbsErrors = 0;
    }

    public void setChild(int parentBranch, FIMTDDNode node) {
    }

    public int getChildIndex(FIMTDDNode child) {
        return -1;
    }

    public int getNumSubtrees() {
        return 1;
    }

    protected boolean skipInLevelCount() {
        return false;
    }

    /**
     * Method to learn from an instance that passes the new instance to the perceptron learner,
     * and also prevents the class value from being truncated to an int when it is passed to the
     * attribute observer
     */
    public void learnFromInstance(Instance inst, boolean growthAllowed) {
        //The prediction must be calculated here -- it may be different from the tree's prediction due to alternate trees

        // Update the statistics for this node
        // number of instances passing through the node
        examplesSeen += inst.weight();

        // sum of y values
        sumOfValues += inst.weight() * inst.classValue();

        // sum of squared y values
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();

        // sum of absolute errors
        sumOfAbsErrors += inst.weight() * Math.abs(tree.normalizeTargetValue(Math.abs(inst.classValue() - getPrediction(inst))));

        if (tree.buildingModelTree()) learningModel.trainOnInstance(inst);

        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
            FIMTDDNumericAttributeClassObserver obs = attributeObservers.get(i);
            if (obs == null) {
                // At this stage all nominal attributes are ignored
                if (inst.attribute(instAttIndex).isNumeric()) {
                    obs = tree.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
            }
            if (obs != null) {
                obs.observeAttributeClass(inst.value(instAttIndex), inst.classValue(), inst.weight());
            }
        }

        if (growthAllowed) {
            checkForSplit(tree);
        }
    }

    /**
     * Return the best split suggestions for this node using the given split criteria
     */
    public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion) {

        List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();

        // Set the nodeStatistics up as the preSplitDistribution, rather than the observedClassDistribution
        double[] nodeSplitDist = new double[]{examplesSeen, sumOfValues, sumOfSquares};

        for (int i = 0; i < this.attributeObservers.size(); i++) {
            FIMTDDNumericAttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs != null) {

                // AT THIS STAGE NON-NUMERIC ATTRIBUTES ARE IGNORED
                AttributeSplitSuggestion bestSuggestion = null;
                if (obs instanceof FIMTDDNumericAttributeClassObserver) {
                    bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion, nodeSplitDist, i, true);
                }

                if (bestSuggestion != null) {
                    bestSuggestions.add(bestSuggestion);
                }
            }
        }
        return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
    }

    /**
     * Retrieve the class votes using the perceptron learner
     */
    public double getPredictionModel(Instance inst) {
        return learningModel.prediction(inst);
    }

    public double getPredictionTargetMean(Instance inst) {
        return (examplesSeen > 0.0) ? sumOfValues / examplesSeen : 0.0;
    }

    public double getPrediction(Instance inst) {
        return (tree.buildingModelTree()) ? getPredictionModel(inst) : getPredictionTargetMean(inst);
    }

    public void checkForSplit(FIMTDD tree) {
        // If it has seen Nmin examples since it was last tested for splitting, attempt a split of this node
        if (examplesSeen - examplesSeenAtLastSplitEvaluation >= tree.gracePeriodOption.getValue()) {
            int index = (parent != null) ? parent.getChildIndex(this) : 0;
            tree.attemptToSplit(this, parent, index);

            // Take note of how many instances were seen when this split evaluation was made, so we know when to perform the next split evaluation
            examplesSeenAtLastSplitEvaluation = examplesSeen;
        }
    }

    public void describeSubtree(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "Leaf ");
        if (tree.buildingModelTree()) {
            learningModel.getModelDescription(out, 0);
        } else {
            out.append(tree.getClassNameString() + " = " + String.format("%.4f", (sumOfValues / examplesSeen)));
            StringUtils.appendNewline(out);
        }
    }

    protected abstract int modelAttIndexToInstanceAttIndex(int i, Instance instance);
}
