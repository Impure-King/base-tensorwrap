## Version 0.0.0.4 Release Notes:

**Major Change**:
1. The SGD Optimizer has been fixed, allowing to implement basic gradient descent.
2. A new AutoGraph Section has been created, in order to start basic graph prototyping.
3. Object exportation has been at under ``python tf.experimental.serialize``, in which python classes and model models can be exported.
4. The Sequential module has been rewritten to boost 200% increase in speed, when compared to TensorFlow.

**Minor Changes**:
1. Expand_dims has been added to allow for easier dimension expansion.
2. An Ops module has been added to allow for manipulate JAX type arrays (which includes ``python tf.Variables`` and ``python tf.range``).
3. The unstable random generating weights has been address, and precompiled inputs has been replaces by dynamic compilation, while retaining speed and reducing error margins.
4. A new self.built is now required for custom layers.

**Current Problems/Gotchas**:
1. Custom Models and Checkpoints are still not supported.
2. Large variation in dense layer units will yield a compressing error.

This version aimed to mostly improve performance and allow for proper training. Additionally, it has now allowed a setup, where future features may be added without errors. However, please note that the internals is bound to change to continue performance increase and introduce more components + flexibility for users.

