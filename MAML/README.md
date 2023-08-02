# MAML Demo

This folder contains scripts for the Model-Agnostic Meta-Learning (MAML) process.

## Files

There are 2 files in this folder:

1. `image_maml.py`: This script implements the MAML algorithm for meta-learning using images as input data.

2. `image_train_model.py`: To train the MAML model, execute this script.

## Training the Model

To train the MAML model, run the `image_train_model.py` script with the following arguments:

```bash
python image_train_model.py --save_dir ./models --model_type metacnn --update_batch_size 128 --test_num_updates 1 --threshold 0 --meta_lr 1e-5 --update_lr 1e-5 --iterations 20000 --gpu_id 3
```

### Arguments:


- `save_dir`: The directory where trained models will be saved. Default is `./models`.

- `model_type`: The type of the model to be used. Default is 'metacnn'.

- `update_batch_size`: The batch size used for inner loop updates during MAML training. Default is 128.

- `test_num_updates`: The number of inner loop updates to be performed during testing. Default is 1.

- `threshold`: A threshold value for some specific purpose (not specified here). Default is 0.

- `meta_lr`: The learning rate for the meta-learner (outer loop) during MAML training. Default is 1e-5.

- `update_lr`: The learning rate for the inner loop updates during MAML training. Default is 1e-5.

- `iterations`: The total number of iterations (steps) for training. Default is 20000.

- `gpu_id`: The ID of the GPU to be used for training. Default is "3".

### Additional Notes:

- The script may have additional arguments commented out, such as `cluster_loss_weight` and `mem_dim`. These can be uncommented and set as needed for specific use cases.

## References

For more information about Model-Agnostic Meta-Learning (MAML), refer to the original paper:

Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *Proceedings of the 34th International Conference on Machine Learning*, 1126–1135. Retrieved from http://proceedings.mlr.press/v70/finn17a/finn17a.pdf

Please note that the information provided in this README assumes you have the necessary dependencies and data properly set up to run the scripts. Make sure to refer to the code documentation for any further details on usage and setup.