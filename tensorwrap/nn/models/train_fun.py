from typing import Any
import tensorwrap as tw
from tensorwrap.module import Module
from tensorwrap.nn import optimizers
import jax
import copy
from termcolor import colored

# Creating a history management class:
class History(Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 1
        self.dict = {}

    def __call__(self, loss_value, metrics_value, val_loss, val_metrics):
        epoch_dict = {
            "loss": loss_value,
            "metrics": metrics_value,
            "val_loss": val_loss,
            "val_metrics": val_metrics
        }
        self.dict[self.epoch] = epoch_dict
        self.epoch += 1
# Creating a Training Class:
class Train(Module):
    def __init__(self, model, loss_fn, optimizer, metric_fn, copy_model = True) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.state = self.optimizer.init(self.model.trainable_variables)
        self.history = History()

        @jax.value_and_grad
        def grad_fn(params, X, y):
            pred = self.model(params, X)
            return self.loss_fn(y, pred)
        self.grad_fn = grad_fn
    
    def train(self, X_train, y_train, epochs = 1, batch_size = 32, validation_data = None, callbacks = [lambda x: x]):
        X_train_batched = tw.experimental.data.Dataset(X_train).batch(batch_size)
        y_train_batched = tw.experimental.data.Dataset(y_train).batch(batch_size)
        if validation_data is not None:
            X_valid, y_valid = validation_data
            compile_val_score = jax.jit(self.val_score)
        else:
            val_loss = None
            val_metrics = None
        compiled_update = jax.jit(self.update)
        self.trainable_variables = [self.model.trainable_variables]
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            for index, (X, y) in enumerate(zip(X_train_batched, y_train_batched)): 
                self.model.trainable_variables, losses, self.state = compiled_update(self.model.trainable_variables, self.state, X, y)
                self.trainable_variables.append(self.model.trainable_variables)
                metrics = self.metric_fn(y, self.model(self.model.trainable_variables, X))
                if validation_data is not None:
                    val_loss, val_metrics = compile_val_score(self.model.trainable_variables, X_valid, y_valid)
                self.loading_animation(X_train_batched.len(), index+1, losses, metrics, val_loss=val_loss, val_metric=val_metrics)
            self.history(losses, metrics, val_loss, val_metrics)
            for callback in callbacks:
                callback(self.history)
            print("\n") 
    
    def return_params(self):
        return self.model.trainable_variables

    def return_model(self):
        return self.model
    
    def return_train_history(self):
        return self.history
    
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

    def loading_animation(self, total_batches, current_batch, loss, metric, val_loss = None, val_metric = None):
        length = 30
        filled_length = int(length * current_batch // total_batches)
        bar = colored('─', "green") * filled_length + '─' * (length - filled_length)
        if val_loss is None:
            val_loss_str = ""
        else:
            val_loss_str = f"    -    val_loss: {val_loss:.5f}"
        
        if val_metric is None:
            val_met_str = ""
        else:
            val_met_str = f"    -    val_metrics: {val_metric:.5f}"
        print(f'\rBatch {current_batch}/{total_batches} [{bar}]    -    loss: {loss:.5f}    -    metric: {metric:.5f}' + val_loss_str + val_met_str, end='', flush=True)

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