import tensorwrap as tw
from tensorwrap.module import Module
from tensorwrap.nn import optimizers
import jax
import copy

# @tw.value_and_grad
# def grad_fn(params, X, y):
#     pred = model(params, X)
#     return loss_fn(y, pred)


# @tw.function
# def update(params, state, X, y):
#     losses, grads = grad_fn(params, X, y)
#     updates, state = optimizer.update(grads, state, params)
#     params = optimizers.apply_updates(params, updates)
#     return params, losses, state

# @tw.function
# def validation(params, X, y):
#     pred = model(params, X)
#     return metrics(y, pred)

# def train(epochs, X_train, y_train, batch_size, model, metric_fn, validation_data = None):
#     global state
#     X_train_batched = tw.experimental.data.Dataset(X_train).batch(batch_size)
#     y_train_batched = tw.experimental.data.Dataset(y_train).batch(batch_size)
#     if validation_data is not None:
#         X_valid, y_valid = validation_data
#     else:
#         val_metrics = "NA"
#     for epoch in range(1, epochs+1):
#         print(f"Epoch {epoch}/{epochs}")
#         for index, (X, y) in enumerate(zip(X_train_batched, y_train_batched)): 
#             model.trainable_variables, losses, state = update(model.trainable_variables, state, X, y)
#             accuracy = metric_fn(y, model(model.trainable_variables, X))
#             if validation_data is not None:
#                 val_metrics = validation(model.trainable_variables, X_valid, y_valid)
#             model.loading_animation(X_train_batched.len(), index+1, losses, accuracy, val_metric=val_metrics)
#         print("\n")    


# Creating a Training Class:
class Train(Module):
    def __init__(self, model, loss_fn, optimizer, metric_fn, copy_model = True) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.state = self.optimizer.init(self.model.trainable_variables)

        @jax.value_and_grad
        def grad_fn(params, X, y):
            pred = self.model(params, X)
            return self.loss_fn(y, pred)
        self.grad_fn = grad_fn
    
    def train(self, X_train, y_train, epochs = 1, batch_size = 32, validation_data = None):
        X_train_batched = tw.experimental.data.Dataset(X_train).batch(batch_size)
        y_train_batched = tw.experimental.data.Dataset(y_train).batch(batch_size)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            compile_val_score = jax.jit(self.val_score)
        else:
            val_loss = None
            val_metrics = None
        compiled_update = jax.jit(self.update)
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            for index, (X, y) in enumerate(zip(X_train_batched, y_train_batched)): 
                self.model.trainable_variables, losses, self.state = compiled_update(self.model.trainable_variables, self.state, X, y)
                accuracy = self.metric_fn(y, self.model(self.model.trainable_variables, X))
                if validation_data is not None:
                    val_loss, val_metrics = compile_val_score(self.model.trainable_variables, X_valid, y_valid)
                self.model.loading_animation(X_train_batched.len(), index+1, losses, accuracy, val_loss=val_loss, val_metric=val_metrics)
            print("\n") 
    
    def return_params(self):
        return self.model.trainable_variables

    def return_model(self):
        return self.model
    
    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, self.loss_fn, self.metric_fn)
    
    def update(self, params, state, X, y):
        losses, grads = self.grad_fn(params, X, y)
        updates, state = self.optimizer.update(grads, state, params)
        params = optimizers.apply_updates(params, updates)
        return params, losses, state
    
    def val_score(self, params, features_valid, labels_valid):
        pred = self.model(params, features_valid)
        val_metrics = self.metric_fn(labels_valid, pred)
        val_loss = self.loss_fn(labels_valid, pred)
        return val_loss, val_metrics