
## Version 0.0.0.6 Prerelease Notes:
**Major Changes**
1. New Layer subclasses have been added: 
    1. ``tensorwrap.nn.layers.Locked`` - Implements a layer identical to the ``tensorwrap.nn.layers.Layer``, but have frozen weights.
    2. ``tensorwrap.nn.layers.Lambda`` - Superclasses layers with no variables, but instead certain functionality.

**Minor Changes**
1. __In Progress__: Adding an experimental feature that converts prebuilt ``tensorwrap.nn.layers.Layer`` objects into ``tensorwrap.nn.layers.Locked`` objects, while preserving state.

<hr>

## Version 0.0.0.5 Release Notes:

**Major Changes**
1. Custom Models have been added, provided they don't use a multi-dim inputs or multi-dim outputs.
2. Speed has been increased 1.5x on average.

**Code Updates**
1. All ``tensorwrap.nn.optimizers.SGD`` objects should now be updated to ``tensorwrap.nn.optimizers.gradient_descent``.

**Current Problems/Gotchas**:
1. Large variation in dense layer units will yield a compressing error.
2. A new ``self.built`` is now required for custom layers.
3. Internal API still iterates layer by layer to implement gradient descent.
4. Custom training loop and batching is currently unavailable.

This version mostly aimed to act as a bug fix and increase speed for users. Additionally, some new modules will be implemented on the next update. Stay tuned for more news.

<hr>

## Version 0.0.0.4 Release Notes:

**Major Changes**:
1. The SGD Optimizer has been fixed, allowing to implement basic gradient descent.
2. A new AutoGraph Section has been created, in order to start basic graph prototyping.
3. Object exportation has been at under ``tensorwrap.experimental.serialize``, in which python classes and model models can be exported.
4. The ``tensorwrap.nn.Sequential`` module has been rewritten to boost 200% increase in speed, when compared to TensorFlow.

**Minor Changes**:
1. Expand_dims has been added to allow for easier dimension expansion.
2. An Ops module has been added to allow for manipulate JAX type arrays (which includes ``tensorwrap.Variables`` and ``tensorwrap.range``).
3. The unstable random generating weights has been address, and precompiled inputs has been replaces by dynamic compilation, while retaining speed and reducing error margins.

**Current Problems/Gotchas**:
1. Custom Models and Checkpoints are still not supported.
2. Large variation in dense layer units will yield a compressing error.
3. A new ``self.built`` is now required for custom layers.
4. Internal API still iterates layer by layer to implement gradient descent.

This version aimed to mostly improve performance and allow for proper training. Additionally, it has now allowed a setup, where future features may be added without errors. However, please note that the internals is bound to change to continue performance increase and introduce more components + flexibility for users.

