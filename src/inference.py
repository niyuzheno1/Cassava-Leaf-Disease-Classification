import torch
import pandas as pd
from config import config, global_params
from src import dataset, transformation, models
import numpy as np
import glob
from typing import Callable, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import wandb

MODEL = global_params.ModelParams()
device = config.DEVICE
test_loader_params = global_params.DataLoaderParams().test_loader


def inference_by_fold(estimator: Callable, state_dicts: List, test_loader: torch.utils.data.DataLoader):
    """Inference the model on the given fold.

    Args:
        estimator (Callable): [description]
        state_dicts (List): [description]
        test_loader (torch.utils.data.DataLoader): [description]

    Returns:
        [type]: [description]
    """
    estimator.to(device)
    estimator.eval()

    probs = []

    with torch.no_grad():
        all_folds_preds = []
        for fold_num, state in enumerate(state_dicts):
            if "model_state_dict" not in state:
                estimator.load_state_dict(state)
            else:
                estimator.load_state_dict(state["model_state_dict"])

            current_fold_preds = []
            for data in tqdm(test_loader, position=0, leave=True):

                images = data["X"].to(device)
                logits = estimator(images)

                sigmoid_preds = logits.sigmoid().detach().cpu().numpy()

                # Need to multiply the predictions by 100 since we divide by 100 in the training script.
                sigmoid_preds = sigmoid_preds * 100

                current_fold_preds.append(sigmoid_preds)

            current_fold_preds = np.concatenate(current_fold_preds, axis=0)
            all_folds_preds.append(current_fold_preds)
        avg_preds = np.mean(all_folds_preds, axis=0)
    return avg_preds


def inference(
    test_df: pd.DataFrame,
    model_dir: str,
    sub_df: pd.DataFrame = None,
):
    """Inference the model on the given fold.

    Dataset and Dataloader are constructed within this function because of TTA.
    """
    # Model, cost function and optimizer instancing

    model = models.PetNeuralNet(pretrained=False).to(device)

    all_preds = {}
    transform_dict = transformation.get_inference_transforms()

    weights = [model_path for model_path in glob.glob(model_dir + "/*.pth")]

    # state_dicts = [torch.load(path)["model_state_dict"] for path in weights]
    state_dicts = [torch.load(path) for path in weights]

    for aug, aug_param in transform_dict.items():
        test_dataset = dataset.PawpularityDataset(df=test_df, transforms=aug_param, mode="test")
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_loader_params)
        predictions = inference_by_fold(estimator=model, state_dicts=state_dicts, test_loader=test_loader)

        all_preds[aug] = predictions

        sub_df["Pawpularity"] = predictions
        sub_df[["Id", "Pawpularity"]].to_csv(f"submission_{aug}.csv", index=False)
        sub_df.head()

        plt.figure(figsize=(12, 6))
        plt.hist(sub_df["Pawpularity"], bins=100)
    return all_preds


def get_grad_cam(model, device, x_tensor, img, label, plot=False):
    result = {"vis": None, "img": None, "pred": None, "label": None}
    with torch.no_grad():
        pred = model(x_tensor.unsqueeze(0).to(device))
    pred = np.concatenate(pred.to("cpu").numpy())[0]
    target_layer = model.model.conv_head
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    output = cam(input_tensor=x_tensor.unsqueeze(0).to(device))
    try:
        vis = show_cam_on_image(img / 255.0, output[0])
    except:
        return result
    if plot:
        fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
        axes[0].imshow(vis)
        axes[0].set_title(f"pred={pred:.4f}")
        axes[1].imshow(img)
        axes[1].set_title(f"target={label}")
        plt.show()
    result = {"vis": vis, "img": img, "pred": pred, "label": label}
    torch.cuda.empty_cache()
    return result


def show_gradcam():
    # wandb_table = wandb.Table(columns=["Id", "Pawpularity", "y_probs", "original_image", "gradcam_image"])

    weights = [
        r"C:\Users\reighns\petfinder\model\tf_efficientnet_b0_ns_best_rmse_fold_1.pt",
        r"C:\Users\reighns\petfinder\model\tf_efficientnet_b0_ns_best_rmse_fold_2.pt",
        r"C:\Users\reighns\petfinder\model\tf_efficientnet_b0_ns_best_rmse_fold_3.pt",
        r"C:\Users\reighns\petfinder\model\tf_efficientnet_b0_ns_best_rmse_fold_4.pt",
        r"C:\Users\reighns\petfinder\model\tf_efficientnet_b0_ns_best_rmse_fold_5.pt",
    ]

    weights = [
        r"C:\Users\reighns\petfinder\model\vit_small_patch16_224_best_rmse_fold_1.pt",
        r"C:\Users\reighns\petfinder\model\vit_small_patch16_224_best_rmse_fold_2.pt",
        r"C:\Users\reighns\petfinder\model\vit_small_patch16_224_best_rmse_fold_3.pt",
        r"C:\Users\reighns\petfinder\model\vit_small_patch16_224_best_rmse_fold_4.pt",
        r"C:\Users\reighns\petfinder\model\vit_small_patch16_224_best_rmse_fold_5.pt",
    ]

    # state_dicts = [torch.load(path)["model_state_dict"] for path in weights]
    state_dicts = [torch.load(path)["model_state_dict"] for path in weights]

    for fold in [1, 2, 3, 4, 5]:
        model = models.PetNeuralNet(pretrained=False).to(device)
        model.load_state_dict(state_dicts[fold - 1])
        model.eval()
        oof_df = pd.read_csv(r"C:\Users\reighns\petfinder\data\processed\oof.csv")
        # oof_df["rmse_diff"] = abs(oof_df["Pawpularity"] - oof_df["0_oof"])
        oof_df = oof_df[oof_df["fold"] == fold].reset_index(drop=True)
        # oof_df = oof_df.sort_values(by="rmse_diff", ascending=True).reset_index(drop=True)

        count = 0
        gradcam_dataset = dataset.PawpularityDataset(
            df=oof_df, transforms=transformation.get_gradcam_transforms(), mode="gradcam"
        )
        for data in gradcam_dataset:
            X, y, original_image, image_id = data["X"], data["y"], data["original_image"], data["image_id"]
            X_unsqueeze = X.unsqueeze(0)

            count += 1

            if count >= 5:
                break

            # def reshape_transform(tensor, height=14, width=14):
            #     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

            #     # Bring the channels to the first dimension,
            #     # like in CNNs.
            #     # result = result.transpose(2, 3).transpose(1, 2)
            #     return result

            if "vit" in MODEL.model_name:
                # blocks[-1].norm1  # for vit models use this, note this is using TIMM backbone.
                target_layers = [model.backbone.blocks[-1].norm1]
            elif "efficientnet" in MODEL.model_name:
                print(model)
                target_layers = [model.backbone.conv_head]
            elif "resnet" in MODEL.model_name:
                target_layers = [model.backbone.layer4[-1]]

            # print(target_layers)
            cam = GradCAM(
                model=model,
                target_layers=target_layers,
                use_cuda=device,
                # reshape_transform=reshape_transform,
            )

            output = cam(input_tensor=X_unsqueeze)  # [32,224,224]
            print(output.shape)
            print(output[0, :].shape)  # [224,224] first image

            original_image = original_image / 255
            original_image = original_image.cpu().detach().numpy()
            assert original_image.shape[-1] == 3, "Channel Last when passing into gradcam."
            print(original_image.shape)

            gradcam_image = show_cam_on_image(original_image, output[0, :], use_rgb=False)

            fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
            axes[0].imshow(original_image)
            # axes[0].set_title(f"pred={pred:.4f}")
            axes[1].imshow(gradcam_image)
            # axes[1].set_title(f"target={label}")
            plt.show()
            torch.cuda.empty_cache()
