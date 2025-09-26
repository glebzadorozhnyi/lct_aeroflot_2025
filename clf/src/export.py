"""Export."""

import os
import time
from dataclasses import dataclass, field
from typing import Type

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from onnxsim import simplify
from tqdm import tqdm

from src.datasets import Batch


@dataclass
class BenchmarkResult:
    ms: float = 0.0


@dataclass
class DiffResult:
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0


@dataclass
class ExportResultItem:
    path: str = ""
    benchmark: BenchmarkResult = field(default_factory=BenchmarkResult)
    diff: DiffResult = field(default_factory=DiffResult)


@dataclass
class ExportResult:
    ts: ExportResultItem = field(default_factory=ExportResultItem)
    onnx: ExportResultItem = field(default_factory=ExportResultItem)


def onnx_benchmark(model_path) -> float:
    """onnx_benchmark"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    total = 0.0
    runs = 100
    input_data = np.zeros((1, *input_shape[1:]), np.uint8)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
    total /= runs
    print(f"Avg: {total:.2f}ms")
    return total


def ts_benchmark(model_path) -> float:
    """ts_benchmark"""
    model = torch.jit.load(model_path, map_location="cuda:0")
    model = model.eval()

    total = 0.0
    runs = 100
    input_data = torch.zeros((1, 640, 640, 3), dtype=torch.uint8).cuda()
    # Warming up
    _ = model(input_data)
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(input_data)
        end = (time.perf_counter() - start) * 1000
        total += end
    total /= runs
    print(f"Avg: {total:.2f}ms")
    return total


def benchmark(model_path) -> float:
    """benchmark"""
    if model_path.endswith(".pt"):
        return ts_benchmark(model_path)
    else:
        return onnx_benchmark(model_path)


def onnx2simp(lm, batch, cfg) -> ExportResultItem:
    """onnx2simp"""
    print("Simplifying ONNX...")
    device = torch.device(cfg.export.onnx.device)
    lm.to(device)
    image = batch.image.cpu().numpy()
    root = os.path.split(os.path.split(cfg.ckpt_path)[0])[0]
    export_path = os.path.join(root, cfg.export.onnx.path)
    export_root = os.path.split(export_path)[0]
    os.makedirs(export_root, exist_ok=True)
    model = onnx.load(export_path)
    model_simp, check = simplify(
        model,
        check_n=2,
        perform_optimization=True,
        skip_fuse_bn=False,
        test_input_shapes={"image": image.shape},
        skipped_optimizers=None,
        skip_constant_folding=False,
        skip_shape_inference=False,
        input_data=None,
        include_subgraph=False,
        unused_output=None,
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save_model(model_simp, export_path)
    print("benchmarking onnx simplified fp32 model...")
    benchmark_ms = benchmark(export_path)
    return ExportResultItem(path=export_path, benchmark=BenchmarkResult(ms=benchmark_ms))


def torch2onnx(lm: LightningModule, batch: Batch, cfg: DictConfig) -> ExportResultItem:
    """torch2onnx"""
    print("Exporting to ONNX...")
    device = torch.device(cfg.export.onnx.device)
    lm.to(device)
    timage = batch.image.to(device)
    root = os.path.split(os.path.split(cfg.ckpt_path)[0])[0]
    export_path = os.path.join(root, cfg.export.onnx.path)
    export_root = os.path.split(export_path)[0]
    os.makedirs(export_root, exist_ok=True)
    torch.onnx.export(
        lm,
        timage,
        export_path,
        export_params=cfg.export.onnx.export_params,
        opset_version=cfg.export.onnx.opset_version,
        do_constant_folding=cfg.export.onnx.do_constant_folding,
        input_names=["image"],
        output_names=["p", "feature"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "p": {0: "batch_size"},
            "feature": {0: "batch_size"},
        },
        verbose=cfg.export.onnx.verbose,
    )
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print("benchmarking onnx fp32 model...")
    benchmark_ms = benchmark(export_path)
    export_result = ExportResultItem(path=export_path, benchmark=BenchmarkResult(ms=benchmark_ms))
    if cfg.export.onnx.simplify:
        export_result = onnx2simp(lm, batch, cfg)
    return export_result


def torch2ts(lm: LightningModule, batch: Batch, cfg: DictConfig) -> ExportResultItem:
    """torch2ts"""
    print("Exporting to TorchScript...")
    device = torch.device(cfg.export.ts.device)
    lm.to(device)
    timage = batch.image.to(device)
    root = os.path.split(os.path.split(cfg.ckpt_path)[0])[0]
    export_path = os.path.join(root, cfg.export.ts.path)
    export_root = os.path.split(export_path)[0]
    os.makedirs(export_root, exist_ok=True)
    lm._jit_is_scripting = True
    ts_model = torch.jit.trace(
        lm,
        timage,
        check_trace=cfg.export.ts.check_trace,
        check_tolerance=cfg.export.ts.check_tolerance,
        strict=cfg.export.ts.strict,
    )
    ts_model.save(export_path)
    print("benchmarking ts model...")
    benchmark_ms = benchmark(export_path)
    if cfg.export.ts.optimize:
        print("Optimizing TorchScript...")
        with torch.jit.optimized_execution(True):
            ts_model = torch.jit.trace(
                lm,
                timage,
                check_trace=cfg.export.ts.check_trace,
                check_tolerance=cfg.export.ts.check_tolerance,
                strict=cfg.export.ts.strict,
            )
        ts_model = torch.jit.optimize_for_inference(ts_model)
        ts_model.save(export_path)

        print("benchmarking ts optimized model...")
        benchmark_ms = benchmark(export_path)
    return ExportResultItem(
        path=export_path,
        benchmark=BenchmarkResult(ms=benchmark_ms),
    )


def predict_torch(lm: LightningModule, batches: list[Batch], device: str) -> np.ndarray:
    """predict_torch"""
    _device = torch.device(device)
    lm.to(_device)
    ps = []
    for batch in tqdm(batches, desc="Predicting torch model", total=len(batches)):
        p, feature = lm(batch.image.to(_device))
        p = p.cpu().numpy()
        ps.append(p)
    ps = np.concatenate(ps, axis=0)
    return ps


def predict_ts(ts_path: str, batches: list[Batch], device: str) -> np.ndarray:
    """predict_ts"""
    _device = torch.device(device)
    model = torch.jit.load(ts_path, map_location=_device).eval()
    ps = []
    for batch in tqdm(batches, desc="Predicting torch model", total=len(batches)):
        p, feature = model(batch.image.to(_device))
        p = p.cpu().numpy()
        ps.append(p)
    ps = np.concatenate(ps, axis=0)
    return ps


def predict_onnx(onnx_path: str, batches: list[Batch], device: str) -> np.ndarray:
    """predict_onnx"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    ps = []
    for batch in tqdm(batches, desc="Predicting onnx model", total=len(batches)):
        p = session.run(["p"], {input_name: batch.image.cpu().numpy()})[0]
        ps.append(p)
    ps = np.concatenate(ps, axis=0)
    return ps


def export(cfg: DictConfig) -> ExportResult:
    """export

    Args:
        cfg (DictConfig): _description_
    """
    # Prepare
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    ldm: LightningDataModule = hydra.utils.instantiate(cfg.ldm)
    lm_class: Type[LightningModule] = hydra.utils.get_class(cfg.lm["_target_"])
    lm = lm_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    lm.eval()
    lm.requires_grad_(False)

    # Get valid batches
    dataloader = ldm.val_dataloader()
    batches = []
    for batch in tqdm(dataloader, desc="Getting valid batches", total=len(dataloader)):
        batches.append(batch)

    # Export
    export_result_ts = torch2ts(lm, batches[0], cfg) if cfg.export.ts.enable else None
    if cfg.export.onnx.enable:
        export_result_onnx = torch2onnx(lm, batches[0], cfg)
    else:
        export_result_onnx = None

    # Check outputs
    # Get Torch outputs
    ps = predict_torch(lm, batches, cfg.export.ts.device)

    # TorchScript
    if cfg.export.ts.enable:
        root = os.path.split(os.path.split(cfg.ckpt_path)[0])[0]
        export_path_ts = os.path.join(root, cfg.export.ts.path)
        ps_ = predict_ts(export_path_ts, batches, cfg.export.ts.device)
        p_diff = np.abs(ps - ps_)
        export_result_onnx.diff.mean = p_diff.mean()
        export_result_onnx.diff.median = np.median(p_diff)
        export_result_onnx.diff.std = np.std(p_diff)
        print("torch vs torchscript", p_diff.mean(), np.median(p_diff), np.std(p_diff))

    # ONNX
    if cfg.export.onnx.enable:
        root = os.path.split(os.path.split(cfg.ckpt_path)[0])[0]
        export_path_fp32 = os.path.join(root, cfg.export.onnx.path)

        # Get ONNX FP32 outputs
        ps_ = predict_onnx(export_path_fp32, batches, cfg.export.onnx.device)
        p_diff = np.abs(ps - ps_)
        export_result_onnx.diff.mean = p_diff.mean()
        export_result_onnx.diff.median = np.median(p_diff)
        export_result_onnx.diff.std = np.std(p_diff)
        print("torch vs onnx-fp32", p_diff.mean(), np.median(p_diff), np.std(p_diff))

    return ExportResult(
        ts=export_result_ts,
        onnx=export_result_onnx,
    )


@hydra.main(config_path="../configs", config_name="cosine-112.yaml", version_base=None)
def main(cfg: DictConfig | None = None):
    """main

    Args:
        cfg (DictConfig | None, optional): _description_. Defaults to None.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))
    result = export(cfg)
    print(result)


if __name__ == "__main__":
    main()
