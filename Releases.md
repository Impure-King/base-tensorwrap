
## Version 0.0.0.6 Prerelease Notes:
**Major Changes**
1. New Layer subclasses have been added: 
    1. ``tensorwrap.nn.layers.Lambda`` - Superclasses layers with no variables, but instead certain functionality.
2. Model speed has been increased:
    1. 60% for cpu devices
    2. 20% for gpu devices.
3. New Layers have been added:
    1. ``tensorwrap.nn.layers.Flatten`` - Returns a flatten input tensor.
    2. ``tensorwrap.nn.layers.Concat`` - concatenates the input tensors and returns one output tensor.
4. New Activation class has been added and ReLU activation is available to use.


**Minor Changes**
1. Jit compilation failure has been addressed.
2. __In Progress:__ An autograph class has been added. It is currently unusable.
3. New JAX built-in methods have been added.
4. Docstrings for some methods have been improved.
5. ``self.built`` for custom layers is no longer required, due to a inheritance fix in the ``tensorwrap.nn.layers.Layer`` module.
6. ``tensorwrap.nn.models.Model.fit()`` now accepts verbose arguments to control output and returns a dictionary with training metrics.

**Current Problems/Gotchas**
1. Internal API still iterates layer by layer to implement gradient descent.
2. Custom training loop and batching is currently unavailable.
3. JSON serializable models are still not available.

This version allows for non-weighted custom layers and has fixed performance/accessibility issues. Additionally, activations have been added and internal api has been cleaned to allow for further development. Please note that the next 
version will continue to change the internals and many namespaces/code may be changed.

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

