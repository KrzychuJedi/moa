package moa.classifiers.trees;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.StringUtils;

import java.io.Serializable;

/**
 * Created by Eduś on 2017-04-27.
 */
public class FIMTDDPerceptron extends AbstractClassifier implements Serializable {

    public FloatOption learningRatioOption = new FloatOption(
            "learningRatio", 'l', "Learning ratio to used for training the Perceptrons in the leaves.",
            0.02, 0, 1.00);

    private static final long serialVersionUID = 1L;

    protected FIMTDD tree;

    // The Perception weights
    protected DoubleVector weightAttribute = new DoubleVector();

    protected double sumOfValues;
    protected double sumOfSquares;

    // The number of instances contributing to this model
    protected double instancesSeen = 0;

    // If the model should be reset or not
    protected boolean reset;

    public String getPurposeString() {
        return "A perceptron regressor as specified by Ikonomovska et al. used for FIMTDD";
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return new double[0];
    }

    @Override
    public void resetLearningImpl() {

    }

    public FIMTDDPerceptron(FIMTDDPerceptron original) {
        this.tree = original.tree;
        weightAttribute = (DoubleVector) original.weightAttribute.copy();
        reset = false;
    }

    public FIMTDDPerceptron(FIMTDD tree) {
        this.tree = tree;
        reset = true;
    }


    public DoubleVector getWeights() {
        return weightAttribute;
    }

    /**
     * Update the model using the provided instance
     */
    public void trainOnInstanceImpl(Instance inst) {

        // Initialize perceptron if necessary
        if (reset == true) {
            reset = false;
            weightAttribute = new DoubleVector();
            instancesSeen = 0;
            for (int j = 0; j < inst.numAttributes(); j++) { // The last index corresponds to the constant b
                weightAttribute.setValue(j, 2 * tree.classifierRandom.nextDouble() - 1);
            }
        }

        // Update attribute statistics
        instancesSeen += inst.weight();

        // Update weights
        double learningRatio = 0.0;
        if (tree.learningRatioConstOption.isSet()) {
            learningRatio = learningRatioOption.getValue();
        } else {
            learningRatio = learningRatioOption.getValue() / (1 + instancesSeen * tree.learningRateDecayFactorOption.getValue());
        }

        sumOfValues += inst.weight() * inst.classValue();
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();

        // Loop for compatibility with bagging methods
        for (int i = 0; i < (int) inst.weight(); i++) {
            updateWeights(inst, learningRatio);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    public void updateWeights(Instance inst, double learningRatio) {
        // Compute the normalized instance and the delta
        DoubleVector normalizedInstance = normalizedInstance(inst);
        double normalizedPrediction = prediction(normalizedInstance);
        double normalizedValue = tree.normalizeTargetValue(inst.classValue());
        double delta = normalizedValue - normalizedPrediction;
        normalizedInstance.scaleValues(delta * learningRatio);

        weightAttribute.addValues(normalizedInstance);
    }

    public DoubleVector normalizedInstance(Instance inst) {
        // Normalize Instance
        DoubleVector normalizedInstance = new DoubleVector();
        for (int j = 0; j < inst.numAttributes() - 1; j++) {
            int instAttIndex = AbstractClassifier.modelAttIndexToInstanceAttIndex(j, inst);
            double mean = tree.sumOfAttrValues.getValue(j) / tree.examplesSeen;
            double sd = computeSD(tree.sumOfAttrSquares.getValue(j), tree.sumOfAttrValues.getValue(j), tree.examplesSeen);
            if (inst.attribute(instAttIndex).isNumeric() && tree.examplesSeen > 1 && sd > 0)
                normalizedInstance.setValue(j, (inst.value(instAttIndex) - mean) / (3 * sd));
            else
                normalizedInstance.setValue(j, 0);
        }
        if (tree.examplesSeen > 1)
            normalizedInstance.setValue(inst.numAttributes() - 1, 1.0); // Value to be multiplied with the constant factor
        else
            normalizedInstance.setValue(inst.numAttributes() - 1, 0.0);
        return normalizedInstance;
    }

    /**
     * Output the prediction made by this perceptron on the given instance
     */
    public double prediction(DoubleVector instanceValues) {
        return scalarProduct(weightAttribute, instanceValues);
    }

    protected double prediction(Instance inst) {
        DoubleVector normalizedInstance = normalizedInstance(inst);
        double normalizedPrediction = prediction(normalizedInstance);
        return denormalizePrediction(normalizedPrediction, tree);
    }

    private double denormalizePrediction(double normalizedPrediction, FIMTDD tree) {
        double mean = tree.sumOfValues / tree.examplesSeen;
        double sd = computeSD(tree.sumOfSquares, tree.sumOfValues, tree.examplesSeen);
        if (getExamplesSeen() > 1)
            return normalizedPrediction * sd * 3 + mean;
        else
            return 0.0;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, getClassNameString() + " =");
        if (getModelContext() != null) {
            for (int j = 0; j < getModelContext().numAttributes() - 1; j++) {
                if (getModelContext().attribute(j).isNumeric()) {
                    out.append((j == 0 || weightAttribute.getValue(j) < 0) ? " " : " + ");
                    out.append(String.format("%.4f", weightAttribute.getValue(j)));
                    out.append(" * ");
                    out.append(getAttributeNameString(j));
                }
            }
            out.append(" + " + weightAttribute.getValue((getModelContext().numAttributes() - 1)));
        }
        StringUtils.appendNewline(out);
    }

    public double computeSD(double squaredVal, double val, double size) {
        if (size > 1)
            return Math.sqrt((squaredVal - ((val * val) / size)) / size);
        else
            return 0.0;
    }

    public double scalarProduct(DoubleVector u, DoubleVector v) {
        double ret = 0.0;
        for (int i = 0; i < Math.max(u.numValues(), v.numValues()); i++) {
            ret += u.getValue(i) * v.getValue(i);
        }
        return ret;
    }

    public abstract double getExamplesSeen();


    @Override
    public boolean isRandomizable() {
        return false;
    }
}
