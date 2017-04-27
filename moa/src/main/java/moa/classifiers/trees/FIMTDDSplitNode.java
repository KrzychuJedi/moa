package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.core.StringUtils;

/**
 * Created by Eduś on 2017-04-27.
 */
public class FIMTDDSplitNode extends FIMTDDInnerNode {

    private static final long serialVersionUID = 1L;

    protected InstanceConditionalTest splitTest;

    /**
     * Create a new SplitNode
     *
     * @param tree
     */
    public FIMTDDSplitNode(InstanceConditionalTest splitTest, FIMTDD tree) {
        super(tree);
        this.splitTest = splitTest;
    }

    public int instanceChildIndex(Instance inst) {
        return splitTest.branchForInstance(inst);
    }

    public FIMTDDNode descendOneStep(Instance inst) {
        return children.get(splitTest.branchForInstance(inst));
    }

    public void describeSubtree(StringBuilder out, int indent) {
        for (int branch = 0; branch < children.size(); branch++) {
            FIMTDDNode child = getChild(branch);
            if (child != null) {
                StringUtils.appendIndented(out, indent, "if ");
                out.append(this.splitTest.describeConditionForBranch(branch,
                        tree.getModelContext()));
                out.append(": ");
                StringUtils.appendNewline(out);
                child.describeSubtree(out, indent + 2);
            }
        }
    }

    public double getPrediction(Instance inst) {
        return children.get(splitTest.branchForInstance(inst)).getPrediction(inst);
    }
}
