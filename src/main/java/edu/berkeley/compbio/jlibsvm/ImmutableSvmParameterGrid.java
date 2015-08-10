package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;

import java.util.Collection;
import java.util.HashSet;

/**
 * Creates a grid of parameters for multiple models, with which to test to SVM with.
 * <p/>
 * For now this supports sweeping over C and different kernels (which may have e.g. different gamma).
 * <p/>
 * To attach new parameters, use the {@link edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid.Builder#build()}
 * method.  Then return the set of parameters by using the {@link ImmutableSvmParameterGrid#getGridParams()}.
 *
 * @param <L> Optional parameter, specifying label type.
 * @param <P> Optional parameter, specifying the type of of parameters.
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ImmutableSvmParameterGrid<L extends Comparable, P> extends ImmutableSvmParameter<L, P> {
  private final Collection<ImmutableSvmParameterPoint<L, P>> gridParams;

  /**
   * Constructor.
   *
   * @param copyFrom
   */
  public ImmutableSvmParameterGrid(Builder<L, P> copyFrom) {
    super(copyFrom);
    gridParams = copyFrom.gridParams;
  }

  /**
   * Public getter.
   *
   * @return
   */
  public Collection<ImmutableSvmParameterPoint<L, P>> getGridParams() {
    return gridParams;
  }

  /**
   * Returns the builder singleton object, used to build a parameter.
   *
   * @param <L> Optional parameter, specifying label type.
   * @param <P> Optional parameter, specifying the type of of parameters.
   * @return
   */
  public static <L extends Comparable, P> Builder<L, P> builder() {
    return new Builder<L, P>();
  }

  /**
   * Returns the builder singleton.
   *
   * @param <L> Optional parameter, specifying label type.
   * @param <P> Optional parameter, specifying the type of of parameters.
   */
  public static class Builder<L extends Comparable, P> extends ImmutableSvmParameter.Builder {
    public Collection<Float> Cset;
    public Collection<KernelFunction<P>> kernelSet;
    private Collection<ImmutableSvmParameterPoint<L, P>> gridParams;

    public Builder(ImmutableSvmParameter.Builder copyFrom) {
      super(copyFrom);

      //default
      Cset = new HashSet<Float>();
      Cset.add(1f);
    }

    /**
     * Copy constructorr.
     *
     * @param copyFrom
     */
    public Builder(ImmutableSvmParameterGrid<L, P> copyFrom) {
      super(copyFrom);

      //default
      //Cset = new HashSet<Float>();
      //Cset.add(1f);

      //Cset = copyFrom.Cset;
      //kernelSet = copyFrom.kernelSet;
      gridParams = copyFrom.gridParams;
    }

    /**
     * Default constructor for singleton.  Do not use.  Instead retrieve
     * instance from
     * {@link ImmutableSvmParameterGrid#builder()}
     */
    public Builder() {
      super();

      //default
      Cset = new HashSet<Float>();
      Cset.add(1f);
    }

    /**
     * Builds one instance of parameters.  Calling this function automatically
     * adds the parameter to the set of parameters.
     *
     * @return
     */

    public ImmutableSvmParameter<L, P> build() {
      ImmutableSvmParameterPoint.Builder<L, P> builder = ImmutableSvmParameterPoint.asBuilder(this);

      if (Cset == null || Cset.isEmpty()) {
        throw new SvmException("Can't build a grid with no C values");
      }

      if (kernelSet == null || kernelSet.isEmpty()) {
        throw new SvmException("Can't build a grid with no kernels");
      }

      if (Cset.size() == 1 && kernelSet.size() == 1) {
        builder.C = Cset.iterator().next();
        builder.kernel = kernelSet.iterator().next();
        return builder.build();
        //	return new ImmutableSvmParameterPoint<L,P>(this);
      }
      gridParams = new HashSet<ImmutableSvmParameterPoint<L, P>>();

      // the C and kernel set here are ignored; we just overwrite them with the grid points

      for (Float gridC : Cset) {
        for (KernelFunction<P> gridKernel : kernelSet) {
          builder.C = gridC;
          builder.kernel = gridKernel;
          builder.gridsearchBinaryMachinesIndependently = false;

          // this copies all the params so we can safely continue modifying the builder
          gridParams.add(builder.build());
        }
      }

      return new ImmutableSvmParameterGrid<L, P>(this);
    }
  }

  /**
   * Cast as a builder object.
   *
   * @return
   */
  public Builder<L, P> asBuilder() {
    return new Builder<L, P>(this);
  }
}
