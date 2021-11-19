
## Prepare Data

```python
df_train, df_test, df_folds, sub = prepare.prepare_data()
```
Returns the dataframes for the training, testing, and folds data, and the submission dataframe.

[]: # Language: python
[]: # Path: prepare.py


## Plot Dataloader

Initialize the dataloader and plot random images.

```python
# Plot random images from dataloader for sanity check.
train_dataset = dataset.CustomDataset(
    df=df_folds,
    transforms=transformation.get_train_transforms(),
    mode="train",
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    # sampler=RandomSampler(dataset_train),
    **loader_params.train_loader,
    worker_init_fn=utils.seed_worker,
)

plot.show_image(
    loader=train_loader,
    nrows=1,
    ncols=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

## Forward Pass

As a sanity check of our model, we can perform a forward pass.

```python
forward_X, forward_y, model_summary = models.forward_pass(
    model=models.CustomNeuralNet()
)
```