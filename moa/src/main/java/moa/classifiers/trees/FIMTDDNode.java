package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.core.SizeOf;
import moa.core.StringUtils;

/**
 * Created by Eduś on 2017-04-27.
 */
public abstract class FIMTDDNode extends AbstractMOAObject {

    private static final long serialVersionUID = 1L;

    public int ID;

    protected FIMTDD tree;

    protected boolean changeDetection = true;

    protected FIMTDDNode parent;

    protected FIMTDDNode alternateTree;
    protected FIMTDDNode originalNode;

    // The statistics for this node:
    // Number of instances that have reached it
    protected double examplesSeen;
    // Sum of y values
    protected double sumOfValues;
    // Sum of squared y values
    protected double sumOfSquares;
    // Sum of absolute errors
    protected double sumOfAbsErrors; // Needed for PH tracking of mean error

    public FIMTDDNode(FIMTDD tree) {
        this.tree = tree;
        ID = tree.maxID;
    }

    public void copyStatistics(FIMTDDNode node) {
        examplesSeen = node.examplesSeen;
        sumOfValues = node.sumOfValues;
        sumOfSquares = node.sumOfSquares;
        sumOfAbsErrors = node.sumOfAbsErrors;
    }

    public int calcByteSize() {
        return (int) SizeOf.fullSizeOf(this);
    }

    /**
     * Set the parent node
     */
    public void setParent(FIMTDDNode parent) {
        this.parent = parent;
    }

    /**
     * Return the parent node
     */
    public FIMTDDNode getParent() {
        return parent;
    }

    public void disableChangeDetection() {
        changeDetection = false;
    }

    public void restartChangeDetection() {
        changeDetection = true;
    }

    public void getDescription(StringBuilder sb, int indent) {

    }

    public double getPrediction(Instance inst) {
        return 0;
    }

    public void describeSubtree(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "Leaf");
    }

    public int getLevel() {
        FIMTDDNode target = this;
        int level = 0;
        while (target.getParent() != null) {
            if (target.skipInLevelCount()) {
                target = target.getParent();
                continue;
            }
            level = level + 1;
            target = target.getParent();
        }
        if (target.originalNode == null) {
            return level;
        } else {
            return level + originalNode.getLevel();
        }
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
}
