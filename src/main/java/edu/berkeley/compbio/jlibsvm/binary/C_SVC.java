package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.qmatrix.BooleanInvertingKernelQMatrix;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 */

/**
 * This is an implementation of the C-Support Vector Classifier for Support Vector Machines (SVM).
 *
 * @param <L> Optional parameter, specifying label type.  See {@link edu.berkeley.compbio.jlibsvm.SVM}
 * @param <P> Optional parameter, specifying the type of of parameters.  See {@link edu.berkeley.compbio.jlibsvm.SVM}
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class C_SVC<L extends Comparable, P> extends BinaryClassificationSVM<L, P> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(C_SVC.class);


// -------------------------- OTHER METHODS --------------------------

  /**
   * Normal training on the entire problem, with no scaling and no cross-validation-based probability measure.
   * <p/>
   * For normal training, you may be interested in {@link edu.berkeley.compbio.jlibsvm.binary.C_SVC#train(BinaryClassificationProblem, edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter)} instead.
   * <p/>
   * See {@link edu.berkeley.compbio.jlibsvm.binary.BinaryClassificationSVM} for more information.
   *
   * @param problem A specified problem.
   * @param Cp
   * @param Cn
   * @param param
   * @return
   */
  @Override
  public BinaryModel<L, P> trainOne(BinaryClassificationProblem<L, P> problem, float Cp, float Cn,
                                    @NotNull ImmutableSvmParameterPoint<L, P> param) {
    float linearTerm = -1f;
    Map<P, Boolean> examples = problem.getBooleanExamples();

    List<SolutionVector<P>> solutionVectors = new ArrayList<SolutionVector<P>>(examples.size());

    for (Map.Entry<P, Boolean> example : examples.entrySet()) {
      SolutionVector<P> sv =
              new SolutionVector<P>(problem.getId(example.getKey()), example.getKey(), example.getValue(),
                      linearTerm);
      //sv.id = problem.getId(example.getKey());
      solutionVectors.add(sv);
    }

    QMatrix<P> qMatrix =
            new BooleanInvertingKernelQMatrix<P>(param.kernel, solutionVectors.size(), param.getCacheRows());

    BinarySolver<L, P> s = new BinarySolver<L, P>(solutionVectors, qMatrix, Cp, Cn, param.eps, param.shrinking);

    BinaryModel<L, P> model = s.solve();
    //	model.vparam = vparam;
    //	model.kernel = kernel;
    model.param = param;
    model.trueLabel = problem.getTrueLabel();
    model.falseLabel = problem.getFalseLabel();
    model.setSvmType(getSvmType());
    model.setScalingModel(problem.getScalingModel());


    //System.err.println(qMatrix.perfString());


    if (Cp == Cn) {
      logger.debug("nu = " + model.getSumAlpha() / (Cp * problem.getNumExamples()));
    }

    for (Map.Entry<P, Double> entry : model.supportVectors.entrySet()) {
      final P key = entry.getKey();
      final Boolean target = examples.get(key);
      if (!target) {
        entry.setValue(entry.getValue() * -1);
      }
    }

    model.compact();

    return model;
  }

  /**
   * Returns the SVM type.
   *
   * @return
   */
  public String getSvmType() {
    return "c_svc";
  }

  @Override
  public void validateParam(@NotNull ImmutableSvmParameter<L, P> param) {
    super.validateParam(param);
    if (param instanceof ImmutableSvmParameterPoint) {
      if (((ImmutableSvmParameterPoint) param).C <= 0) {
        throw new SvmException("C <= 0");
      }
    }
  }
}
