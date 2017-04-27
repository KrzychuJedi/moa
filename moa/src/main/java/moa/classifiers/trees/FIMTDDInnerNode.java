package moa.classifiers.trees;

import moa.core.AutoExpandVector;

/**
 * Created by Eduś on 2017-04-27.
 */
public abstract class FIMTDDInnerNode extends FIMTDDNode {
    // The InnerNode and SplitNode design is used for easy extension in ORTO
    private static final long serialVersionUID = 1L;

    protected AutoExpandVector<FIMTDDNode> children = new AutoExpandVector<FIMTDDNode>();

    // The error values for the Page Hinckley test
    // PHmT = the cumulative sum of the errors
    // PHMT = the minimum error value seen so far
    protected double PHsum = 0;
    protected double PHmin = Double.MAX_VALUE;

    // Keep track of the statistics for loss error calculations
    protected double lossExamplesSeen;
    protected double lossFadedSumOriginal;
    protected double lossFadedSumAlternate;
    protected double lossNumQiTests;
    protected double lossSumQi;
    protected double previousWeight = 0;

    public FIMTDDInnerNode(FIMTDD tree) {
        super(tree);
    }

    public int numChildren() {
        return children.size();
    }

    public FIMTDDNode getChild(int index) {
        return children.get(index);
    }

    public int getChildIndex(FIMTDDNode child) {
        return children.indexOf(child);
    }

    public void setChild(int index, FIMTDDNode child) {
        children.set(index, child);
    }

    public void disableChangeDetection() {
        changeDetection = false;
        for (FIMTDDNode child : children) {
            child.disableChangeDetection();
        }
    }

    public void restartChangeDetection() {
        if (alternateTree == null) {
            changeDetection = true;
            PHsum = 0;
            PHmin = Integer.MAX_VALUE;
            for (FIMTDDNode child : children)
                child.restartChangeDetection();
        }
    }

    /**
     * Check to see if the tree needs updating
     */
    public boolean PageHinckleyTest(double error, double threshold) {
        // Update the cumulative mT sum
        PHsum += error;

        // Update the minimum mT value if the new mT is
        // smaller than the current minimum
        if (PHsum < PHmin) {
            PHmin = PHsum;
        }
        // Return true if the cumulative value - the current minimum is
        // greater than the current threshold (in which case we should adapt)
        return PHsum - PHmin > threshold;
    }

    public void initializeAlternateTree() {
        // Start a new alternate tree, beginning with a learning node
        alternateTree = tree.newLeafNode();
        alternateTree.originalNode = this;

        // Set up the blank statistics
        // Number of instances reaching this node since the alternate tree was started
        lossExamplesSeen = 0;
        // Faded squared error (original tree)
        lossFadedSumOriginal = 0;
        // Faded squared error (alternate tree)
        lossFadedSumAlternate = 0;
        // Number of evaluations of alternate tree
        lossNumQiTests = 0;
        // Sum of Qi values
        lossSumQi = 0;
        // Number of examples at last test
        previousWeight = 0;

        // Disable the change detection mechanism bellow this node
        disableChangeDetection();
    }
}
