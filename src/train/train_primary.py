import argparse
import inspect
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from architectures.robot_encoder import RobotEncoder
from architectures.predictor import PredictorNetwork
from datasets.robot.primary_dataset import (
    RobotPrimaryDataset,
    compute_train_action_stats,
    compute_train_state_stats,
    normalize_robot_images,
)
from losses import SIGReg
from primary_contract import (
    ACTION_ENCODING_ID,
    ARCHITECTURE_ID,
    CHECKPOINT_FORMAT_VERSION,
    IMAGE_NORMALIZATION_ID,
)
from utils import amp_dtype_label, autocast_context


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _latest_epoch_checkpoint(save_dir: Path):
    checkpoints = []
    for path in save_dir.glob("epoch_*_model.zip"):
        match = re.fullmatch(r"epoch_(\d+)_model\.zip", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None), key=lambda item: item[0])


def _load_training_state(
    path: Path, encoder, predictor, optimizer, scaler, device, expected_metadata
):
    with zipfile.ZipFile(path, "r") as zf, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if "training_state.pth" not in zf.namelist():
            raise ValueError(f"{path} has no resumable training state")
        zf.extract("training_state.pth", tmp)
        try:
            training_state = torch.load(
                tmp / "training_state.pth", map_location=device, weights_only=False
            )
        except TypeError:
            training_state = torch.load(tmp / "training_state.pth", map_location=device)

        for key, expected in expected_metadata.items():
            actual = training_state.get(key)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint {key}={actual!r}, expected {expected!r}. "
                    "Start a new run with the corrected architecture and action contract."
                )

        zf.extract("encoder_weights.pth", tmp)
        zf.extract("predictor_weights.pth", tmp)
        encoder.load_state_dict(
            torch.load(tmp / "encoder_weights.pth", map_location=device, weights_only=True)
        )
        predictor.load_state_dict(
            torch.load(tmp / "predictor_weights.pth", map_location=device, weights_only=True)
        )

        try:
            optimizer.load_state_dict(training_state["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "The checkpoint optimizer groups do not match the current optimizer. "
                "Start this run from scratch."
            ) from exc
        if training_state.get("scaler"):
            scaler.load_state_dict(training_state["scaler"])

    return training_state


def _save_checkpoint(
    zip_path: Path,
    encoder,
    predictor,
    optimizer,
    scaler,
    epoch: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    checkpoint_metadata: dict,
    results: dict,
    config: dict | None,
    best_val_loss: float,
    best_epoch: int,
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        encoder_path = tmp / "encoder_weights.pth"
        predictor_path = tmp / "predictor_weights.pth"
        training_path = tmp / "training_state.pth"
        state_stats_path = tmp / "state_stats.pth"

        torch.save(encoder.state_dict(), encoder_path)
        torch.save(predictor.state_dict(), predictor_path)
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "state_mean": state_mean.cpu(),
                "state_std": state_std.cpu(),
                "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(),
                "action_low": action_low.cpu(),
                "action_high": action_high.cpu(),
                **checkpoint_metadata,
                "results": results,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            },
            training_path,
        )
        torch.save(
            {
                "state_mean": state_mean.cpu(),
                "state_std": state_std.cpu(),
                "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(),
                "action_low": action_low.cpu(),
                "action_high": action_high.cpu(),
                **checkpoint_metadata,
            },
            state_stats_path,
        )

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(encoder_path, "encoder_weights.pth")
            zf.write(predictor_path, "predictor_weights.pth")
            zf.write(training_path, "training_state.pth")
            zf.write(state_stats_path, "state_stats.pth")
            if config is not None:
                zf.writestr("config.yaml", yaml.safe_dump(config, sort_keys=False))


def _atomic_copy(source: Path, destination: Path):
    pending = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, pending)
    pending.replace(destination)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_optimizer(
    named_parameters,
    lr,
    betas,
    weight_decay,
    on_cuda,
    no_decay_parameters,
):
    no_decay_parameter_ids = {id(parameter) for parameter in no_decay_parameters}
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        # Only BatchNorm affine parameters avoid decay. LayerNorm scales and
        # ordinary biases need the original decay to prevent the ViT runaway
        # observed when every one-dimensional parameter was exempted.
        if id(parameter) in no_decay_parameter_ids:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    kwargs = {
        "lr": lr,
        "betas": betas,
        "weight_decay": 0.0,
    }
    if on_cuda and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        **kwargs,
    )
    print(
        f"AdamW parameter groups: {len(decay_parameters)} decayed tensors, "
        f"{len(no_decay_parameters)} BatchNorm affine tensors without decay"
    )
    return optimizer


def _batch_norm_modules(*models):
    return [
        module
        for model in models
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


@torch.no_grad()
def _recalibrate_batch_norm(
    encoder,
    predictor,
    dataloader,
    max_batches,
    device,
    non_blocking,
    amp_dtype,
):
    batch_norms = _batch_norm_modules(encoder, predictor)
    if max_batches <= 0 or not batch_norms:
        encoder.eval()
        predictor.eval()
        return 0

    encoder.eval()
    predictor.eval()
    momenta = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.reset_running_stats()
        # Use a cumulative average during calibration so the result does not
        # depend disproportionately on the final few shuffled batches.
        module.momentum = None
        module.train()

    batches_seen = 0
    try:
        for batch in dataloader:
            if batches_seen >= max_batches:
                break
            above, wrist, state, actions = _prepare_batch(
                batch, device, non_blocking
            )
            with autocast_context(device, amp_dtype):
                embeddings = encoder(above, wrist, state)
                predictor(embeddings[:, :-1], actions)
            batches_seen += 1
    finally:
        for module, momentum in zip(batch_norms, momenta):
            module.momentum = momentum
        encoder.eval()
        predictor.eval()

    return batches_seen


def _batch_norm_diagnostics(model):
    batch_norms = _batch_norm_modules(model)
    if not batch_norms:
        return float("nan"), float("nan"), float("nan")

    scales = torch.cat(
        [module.weight.detach().float().flatten() for module in batch_norms]
    )
    running_vars = torch.cat(
        [module.running_var.detach().float().flatten() for module in batch_norms]
    )
    return (
        scales.square().mean().sqrt().item(),
        running_vars.min().item(),
        running_vars.max().item(),
    )


def _loader_options(num_workers, persistent_workers, pin_memory,
                    prefetch_factor, in_order, timeout, start_method):
    options = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers and num_workers > 0,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["timeout"] = timeout
        options["multiprocessing_context"] = start_method
    if "in_order" in inspect.signature(DataLoader).parameters:
        options["in_order"] = in_order
    return options


def _prepare_batch(batch, device, non_blocking):
    above, wrist, state, actions = batch
    # Keep images as uint8 in workers and pinned host memory. Conversion on the
    # GPU cuts host memory and transfer traffic by 4x.
    above = above.to(device, non_blocking=non_blocking).float().mul_(1.0 / 255.0)
    wrist = wrist.to(device, non_blocking=non_blocking).float().mul_(1.0 / 255.0)
    above = normalize_robot_images(above)
    wrist = normalize_robot_images(wrist)
    state = state.to(device, non_blocking=non_blocking)
    actions = actions.to(device, non_blocking=non_blocking)
    if actions.ndim != 4:
        raise ValueError(f"expected ordered action blocks [B, W, S, A], got {actions.shape}")
    actions = actions.flatten(start_dim=2)
    return above, wrist, state, actions


def train(device,
          data_dir: Path,
          log_dir: Path,
          save_dir: Path,
          frame_skip: int,
          lr: float,
          seed: int,
          lambd: float,
          epochs: int,
          batch_size: int,
          num_workers: int,
          persistent_workers: bool,
          pin_memory: bool,
          prefetch_factor: int,
          in_order: bool,
          loader_timeout: float,
          loader_start_method: str,
          compile_model: bool,
          window_size: int,
          embedding_dim: int,
          action_dim: int,
          pred_nheads: int,
          dropout: float,
          pred_nlayers: int,
          pred_dim_head: int,
          pred_mlp_dim: int,
          action_smoothing_dim: int,
          projector_hidden_dim: int,
          patch_size: int,
          enc_nheads: int,
          enc_nlayers: int,
          enc_npatches: int,
          knots: int,
          num_proj: int,
          optimizer_betas: tuple[float, float],
          optimizer_weight_decay: float,
          gradient_clip_norm: float,
          bn_calibration_batches: int,
          amp_dtype: str,
          resume: bool = False,
          config: dict | None = None):

    torch.manual_seed(seed)
    if isinstance(device, str):
        on_cuda = device.startswith("cuda")
    else:
        on_cuda = device.type == "cuda"

    if on_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate these once from the train mask, then use the same values for
    # both splits and save them with every checkpoint.
    h5_path = Path(data_dir) / "robot" / "mimicgen" / "demo.hdf5"
    state_mean, state_std = compute_train_state_stats(h5_path)
    action_mean, action_std, action_low, action_high = compute_train_action_stats(h5_path)
    print(f"state mean: {state_mean.tolist()}")
    print(f"state std:  {state_std.tolist()}")
    print(f"action mean: {action_mean.tolist()}")
    print(f"action std:  {action_std.tolist()}")
    print(
        f"temporal stride: frame_skip={frame_skip} "
        f"({frame_skip} simulator steps per model transition)"
    )

    train_dataset = RobotPrimaryDataset(
        data_dir=data_dir,
        frame_skip=frame_skip,
        window=window_size,
        val=False,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_dataset = RobotPrimaryDataset(
        data_dir=data_dir,
        frame_skip=frame_skip,
        window=window_size,
        val=True,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    predictor_action_dim = frame_skip * action_dim
    checkpoint_metadata = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_id": ARCHITECTURE_ID,
        "action_encoding_id": ACTION_ENCODING_ID,
        "image_normalization_id": IMAGE_NORMALIZATION_ID,
        "frame_skip": frame_skip,
        "primitive_action_dim": action_dim,
        "predictor_action_dim": predictor_action_dim,
    }

    loader_options = _loader_options(
        num_workers,
        persistent_workers,
        pin_memory,
        prefetch_factor,
        in_order,
        loader_timeout,
        loader_start_method,
    )
    print(
        f"DataLoader: workers={num_workers}, start_method="
        f"{loader_options.get('multiprocessing_context', 'single-process')}, "
        f"prefetch={loader_options.get('prefetch_factor', 0)}, "
        f"persistent={loader_options['persistent_workers']}"
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )

    predictor = PredictorNetwork(
        d_model=embedding_dim,
        nheads=pred_nheads,
        d_action=predictor_action_dim,
        dropout=dropout,
        nlayers=pred_nlayers,
        window=window_size,
        dim_head=pred_dim_head,
        mlp_dim=pred_mlp_dim,
        action_smoothing_dim=action_smoothing_dim,
        projector_hidden_dim=projector_hidden_dim,
    ).to(device)
    encoder = RobotEncoder(
        patch=patch_size,
        nheads=enc_nheads,
        nlayers=enc_nlayers,
        d_model=embedding_dim,
        npatches=enc_npatches,
        projector_hidden_dim=projector_hidden_dim,
    ).to(device)

    gpu_name = torch.cuda.get_device_name(0) if on_cuda else "cpu"
    amp_label = amp_dtype_label(amp_dtype, device)
    print(f"networks are on device: {device} ({gpu_name})")
    print(f"AMP: {amp_label} (config amp_dtype={amp_dtype})")

    sig_reg = SIGReg(knots=knots, num_proj=num_proj).to(device)
    pred_loss_fn = nn.MSELoss()
    named_parameters = [
        (f"predictor.{name}", parameter)
        for name, parameter in predictor.named_parameters()
    ] + [
        (f"encoder.{name}", parameter)
        for name, parameter in encoder.named_parameters()
    ]
    trainable_parameters = [
        parameter for _, parameter in named_parameters if parameter.requires_grad
    ]
    batch_norm_parameters = [
        parameter
        for module in _batch_norm_modules(predictor, encoder)
        for parameter in module.parameters(recurse=False)
        if parameter.requires_grad
    ]
    optimizer = _make_optimizer(
        named_parameters,
        lr,
        optimizer_betas,
        optimizer_weight_decay,
        on_cuda,
        batch_norm_parameters,
    )
    scaler = _make_grad_scaler(enabled=on_cuda and amp_label == "fp16")

    writer = SummaryWriter(log_dir=log_dir)
    results = {
        "train_loss": [],
        "train_prediction_loss": [],
        "train_sigreg_loss": [],
        "train_grad_norm": [],
        "val_loss": [],
        "val_prediction_loss": [],
        "val_sigreg_loss": [],
        "encoder_bn_gamma_rms": [],
        "encoder_bn_running_var_min": [],
        "encoder_bn_running_var_max": [],
    }
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1

    if resume:
        last_epoch, checkpoint_path = _latest_epoch_checkpoint(save_dir)
        if checkpoint_path is None:
            print(f"No epoch_*_model.zip found in {save_dir}; starting from scratch")
        else:
            training_state = _load_training_state(
                checkpoint_path,
                encoder,
                predictor,
                optimizer,
                scaler,
                device,
                checkpoint_metadata,
            )
            if not training_state:
                raise ValueError(
                    f"{checkpoint_path} contains weights only and cannot resume "
                    "optimizer state. Start a new run or use a new-format checkpoint."
                )
            saved_mean = training_state["state_mean"].cpu()
            saved_std = training_state["state_std"].cpu()
            if not torch.allclose(saved_mean, state_mean) or not torch.allclose(saved_std, state_std):
                raise ValueError("Dataset state statistics do not match the checkpoint")
            for key, current in (
                ("action_mean", action_mean),
                ("action_std", action_std),
                ("action_low", action_low),
                ("action_high", action_high),
            ):
                saved = training_state.get(key)
                if saved is None or not torch.allclose(saved.cpu(), current):
                    raise ValueError(f"Dataset {key} does not match the checkpoint")
            results = training_state.get("results", results)
            start_epoch = int(training_state.get("epoch", last_epoch)) + 1
            previous_val_losses = results.get("val_loss", [])
            best_val_loss = float(
                training_state.get(
                    "best_val_loss",
                    min(previous_val_losses, default=float("inf")),
                )
            )
            best_epoch = int(training_state.get("best_epoch", -1))
            if best_epoch < 0 and previous_val_losses:
                best_epoch = min(
                    range(len(previous_val_losses)),
                    key=previous_val_losses.__getitem__,
                )
            print(f"resumed {checkpoint_path.name}; continuing at epoch {start_epoch}")

    if compile_model and on_cuda:
        if hasattr(encoder, "compile") and hasattr(predictor, "compile"):
            encoder.compile()
            predictor.compile()
            print("torch.compile enabled for encoder and predictor")
        else:
            print("torch.compile requested but unavailable; using eager mode")

    non_blocking = on_cuda and pin_memory

    print(f"Training epochs [{start_epoch}, {start_epoch + epochs})")
    for epoch in tqdm(range(start_epoch, start_epoch + epochs)):
        print(f"Epoch: {epoch}\n------------")

        train_metrics = torch.zeros(4, device=device)

        encoder.train()
        predictor.train()
        for batch in train_dataloader:
            above, wrist, state, actions = _prepare_batch(
                batch, device, non_blocking
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype):
                # Encode o_t ... o_{t+window} once, then use the same shifted
                # prediction objective as the original LeWM training process.
                all_embeddings = encoder(above, wrist, state)
                pred_embeddings = predictor(all_embeddings[:, :-1], actions)
                target_embeddings = all_embeddings[:, 1:]

                pred_loss = pred_loss_fn(pred_embeddings, target_embeddings)
                reg_loss = sig_reg(all_embeddings.transpose(0, 1))
                loss = pred_loss + lambd * reg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()

            train_metrics += torch.stack(
                (
                    loss.detach(),
                    pred_loss.detach(),
                    reg_loss.detach(),
                    grad_norm.detach(),
                )
            )

        num_train_batches = len(train_dataloader)
        train_total, train_prediction, train_sigreg, train_grad_norm = (
            train_metrics.div(num_train_batches).cpu().tolist()
        )

        calibration_batches_seen = _recalibrate_batch_norm(
            encoder=encoder,
            predictor=predictor,
            dataloader=train_dataloader,
            max_batches=bn_calibration_batches,
            device=device,
            non_blocking=non_blocking,
            amp_dtype=amp_dtype,
        )
        bn_gamma_rms, bn_var_min, bn_var_max = _batch_norm_diagnostics(encoder)

        val_metrics = torch.zeros(3, device=device)

        encoder.eval()
        predictor.eval()
        with torch.inference_mode():
            for batch in val_dataloader:
                above, wrist, state, actions = _prepare_batch(
                    batch, device, non_blocking
                )

                with autocast_context(device, amp_dtype):
                    all_embeddings = encoder(above, wrist, state)
                    pred_embeddings = predictor(all_embeddings[:, :-1], actions)
                    target_embeddings = all_embeddings[:, 1:]

                    pred_loss = pred_loss_fn(pred_embeddings, target_embeddings)
                    reg_loss = sig_reg(all_embeddings.transpose(0, 1))
                    loss = pred_loss + lambd * reg_loss

                val_metrics += torch.stack((loss, pred_loss, reg_loss))

        num_val_batches = len(val_dataloader)
        val_total, val_prediction, val_sigreg = (
            val_metrics.div(num_val_batches).cpu().tolist()
        )

        results["train_loss"].append(train_total)
        results["train_prediction_loss"].append(train_prediction)
        results["train_sigreg_loss"].append(train_sigreg)
        results.setdefault("train_grad_norm", []).append(train_grad_norm)
        results["val_loss"].append(val_total)
        results["val_prediction_loss"].append(val_prediction)
        results["val_sigreg_loss"].append(val_sigreg)
        results.setdefault("encoder_bn_gamma_rms", []).append(bn_gamma_rms)
        results.setdefault("encoder_bn_running_var_min", []).append(bn_var_min)
        results.setdefault("encoder_bn_running_var_max", []).append(bn_var_max)

        writer.add_scalars(
            main_tag="Loss/Total",
            tag_scalar_dict={"train": train_total, "validation": val_total},
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Loss/Prediction",
            tag_scalar_dict={"train": train_prediction, "validation": val_prediction},
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Loss/SIGReg",
            tag_scalar_dict={"train": train_sigreg, "validation": val_sigreg},
            global_step=epoch,
        )
        writer.add_scalar("Optimization/GradientNorm", train_grad_norm, epoch)
        writer.add_scalar("BatchNorm/EncoderGammaRMS", bn_gamma_rms, epoch)
        writer.add_scalar("BatchNorm/EncoderRunningVarMin", bn_var_min, epoch)
        writer.add_scalar("BatchNorm/EncoderRunningVarMax", bn_var_max, epoch)

        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total
            best_epoch = epoch

        zip_path = save_dir / f"epoch_{epoch}_model.zip"
        _save_checkpoint(
            zip_path=zip_path,
            encoder=encoder,
            predictor=predictor,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            action_low=action_low,
            action_high=action_high,
            checkpoint_metadata=checkpoint_metadata,
            results=results,
            config=config,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
        if is_best:
            _atomic_copy(zip_path, save_dir / "best_model.zip")

        print(
            f"Train total: {train_total:.4f} "
            f"(prediction {train_prediction:.4f}, SIGReg {train_sigreg:.4f}) | "
            f"Validation total: {val_total:.4f} "
            f"(prediction {val_prediction:.4f}, SIGReg {val_sigreg:.4f}) | "
            f"grad norm {train_grad_norm:.4f} | "
            f"encoder BN gamma RMS {bn_gamma_rms:.4f} "
            f"(recalibrated on {calibration_batches_seen} batches)"
        )
        if is_best:
            print(f"new best validation checkpoint → {save_dir / 'best_model.zip'}")

    train_dataset.close()
    val_dataset.close()
    writer.close()
    return results


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default_primary.yaml",
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run. Checkpoints save to <save_dir>/<run_name>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer, and AMP state from the latest epoch checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    paths = config["paths"]
    training = config["training"]
    model = config["model"]
    loss = config["loss"]
    optimizer_cfg = config["optimizer"]

    data_dir = PROJECT_DIR / paths["data_dir"]
    log_dir = PROJECT_DIR / paths["log_dir"] / args.run_name
    save_dir = PROJECT_DIR / paths["save_dir"] / args.run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = train(
        device=device,
        data_dir=data_dir,
        log_dir=log_dir,
        save_dir=save_dir,
        frame_skip=training["frame_skip"],
        lr=training["lr"],
        seed=training["seed"],
        lambd=training["lambd"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        persistent_workers=training["persistent_workers"],
        pin_memory=training["pin_memory"],
        prefetch_factor=training.get("prefetch_factor", 2),
        in_order=training.get("in_order", True),
        loader_timeout=training.get("loader_timeout", 120),
        loader_start_method=training.get("loader_start_method", "spawn"),
        compile_model=training.get("compile_model", True),
        window_size=training["window_size"],
        embedding_dim=model["embedding_dim"],
        action_dim=model["action_dim"],
        pred_nheads=model["pred_nheads"],
        dropout=model["dropout"],
        pred_nlayers=model["pred_nlayers"],
        pred_dim_head=model["pred_dim_head"],
        pred_mlp_dim=model["pred_mlp_dim"],
        action_smoothing_dim=model["action_smoothing_dim"],
        projector_hidden_dim=model["projector_hidden_dim"],
        patch_size=model["patch_size"],
        enc_nheads=model["enc_nheads"],
        enc_nlayers=model["enc_nlayers"],
        enc_npatches=model["enc_npatches"],
        knots=loss["knots"],
        num_proj=loss["num_proj"],
        optimizer_betas=tuple(optimizer_cfg["betas"]),
        optimizer_weight_decay=optimizer_cfg["weight_decay"],
        gradient_clip_norm=training.get("gradient_clip_norm", 1.0),
        bn_calibration_batches=training.get("bn_calibration_batches", 256),
        amp_dtype=training.get("amp_dtype", "auto"),
        resume=args.resume,
        config=config,
    )
