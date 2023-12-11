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
        self.decipher_dict = {
            "loss": 0,
            "metrics": 1,
            "val_loss": 2,
            "val_metrics": 3
        }
    
    def get(self, name):
        if name == "epochs":
            return jax.numpy.array(list(self.dict.keys()))
        return jax.numpy.array(list(self.dict.values()))[:, self.decipher_dict[name]]

    def __call__(self, loss_value, metrics_value, val_loss, val_metrics):
        epoch_list = [
            loss_value,
            metrics_value,
            val_loss,
            val_metrics
        ]
        self.dict[self.epoch] = epoch_list
        self.epoch += 1

# Creating a Training Class:
class Train(Module):
    def __init__(self, model, loss_fn, optimizer, metric_fn, copy_model = True) -> None:
        super().__init__()
        # Getting the necessary metrics:
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        
        if copy_model:
            self.model = copy.deepcopy(model)
        else:
            self.model = model
            
        self.optimizer = optimizer
        self.state = self.optimizer.init(self.model.params)

        # Loading the History object:
        self.history = History()

        # Creating a grad_fn
        def grad_fn(params, X, y):
            pred = self.model(params, X)
            return self.loss_fn(y, pred), pred
        
        self.grad_fn = jax.value_and_grad(grad_fn, has_aux=True)

    
    def fit(self, X_train, y_train, epochs = 1, batch_size = 32, validation_data = None, callbacks = [lambda x: x]):
        # Batching the data as requested:
        X_train_batched = tw.experimental.Dataloader(X_train).batch(batch_size).shuffle(1)
        y_train_batched = tw.experimental.Dataloader(y_train).batch(batch_size).shuffle(1)
        
        # Checking if validation is needed:
        if validation_data is not None:
            X_valid, y_valid = validation_data
            compile_val_score = jax.jit(self.val_score)
        else:
            val_loss = None
            val_metrics = None
        
        # Compiling update function:
        compiled_update = jax.jit(self.update)

        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            # Resetting the metric:
            # self.metric_fn.reset()
            
            # Looping through the batched data:
            for index, (X, y) in enumerate(zip(X_train_batched, y_train_batched)):
                # Performing 1 update:
                self.model.params, losses, self.state, pred = compiled_update(self.model.params, self.state, X, y)

                # Gathering training metrics:
                metrics = self.metric_fn(y, pred)

                # Showing loading animation:
                self.loading_animation(X_train_batched.len(), index+1, losses, metrics)
            
            # Validating if needed:
            if validation_data:
                    val_loss, val_metrics = compile_val_score(self.model.params, X_valid, y_valid)
            
            self.loading_animation(X_train_batched.len(), index+1, losses, metrics, val_loss=val_loss, val_metric=val_metrics)
            self.history(losses, metrics, val_loss, val_metrics) # Storing the metrics.
            
            # Performing callbacks:
            for callback in callbacks:
                callback(self.history)
            print("\n")
    

    def get_params(self):
        return self.model.params

    def get_model(self):
        return self.model
    
    def get_train_history(self):
        return self.history
    
    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, self.loss_fn, self.metric_fn)
    
    def update(self, params, state, X, y):
        # Performing Backward Pass:
        (losses, pred), grads = self.grad_fn(params, X, y)

        # Optimizing:
        updates, state = self.optimizer.update(grads, state, params)
        params = optimizers.apply_updates(params, updates)
        
        return params, losses, state, pred
    
    def val_score(self, params, features_valid, labels_valid):
        pred = self.model(params, features_valid)
        val_metrics = self.metric_fn(labels_valid, pred)
        val_loss = self.loss_fn(labels_valid, pred)
        return val_loss, val_metrics

    def loading_animation(self, total_batches, current_batch, loss, metric, val_loss = None, val_metric = None):
        length = 30
        filled_length = int(length * current_batch // total_batches)
        bar = colored('─', "green") * filled_length + colored('─', "yellow") * (length - filled_length)
        if val_loss is None:
            val_loss_str = ""
        else:
            val_loss_str = f" - val_loss: {val_loss:.4f}"
        
        if val_metric is None:
            val_met_str = ""
        else:
            val_met_str = f" - val_metrics: {val_metric:.4f}"
        print(f'\rBatch {current_batch}/{total_batches} {bar} - loss: {loss:.4f} - metric: {metric:.4f}' + val_loss_str + val_met_str, end='', flush=True)
    

def basic_train_code():
    """Returns a basic training loop for reference."""
    loop = """
    # Creating a train function:
    from functools import partial
    loss_fn = nn.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = nn.optimizers.adam(learning_rate=1e-3)
    state = optimizer.init(model.params)
    metric_fn = nn.metrics.Accuracy()

    @partial(tw.value_and_grad, has_aux=True)
    def predict(params, X, y):
        pred = model(params, X)
        return loss_fn(y, pred), pred

    @tw.function
    def update(params, state, X, y):
        (losses, pred), grads = predict(params, X, y)
        updates, state = optimizer.update(grads, state, params)
        params = nn.optimizers.apply_updates(params, updates)
        return params, losses, state, pred

    def train():
        global state, model
        for i in range(3):
            print(f"Epoch {i}")
            for X, y in zip(X_train_batched, y_train_batched):
                model.params, loss, state, pred = update(model.params, state, X, y)
                metrics = metric_fn(y, pred)
                print(f"\rLoss: {loss:.5f} \t\t\t Metrics: {metrics:.5f}", end='', sep='')
            print('\n')

"""
    print(loop)