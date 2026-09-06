"""Convert SenseVoiceSmall CTC head to OpenVINO IR for NPU inference.

The CTC head (ctc.ctc_lo: Linear 512->25055, 12.8M params) is a fixed-shape
linear projection -- no autoregression, no KV cache, no dynamic sequence.
This is exactly the operation type NPU handles well.

Output: models/SenseVoiceSmall_ov/ctc_head.xml + ctc_head.bin
"""

import os
import sys
import time
import numpy as np

MODEL_DIR = "D:/APPs/OpenVINO/demo/local-meeting-minutes/models/SenseVoiceSmall"
OUTPUT_DIR = "D:/APPs/Intel苏州线下比赛/voiceguard/models/SenseVoiceSmall_ov"

print("=== SenseVoice CTC Head OpenVINO Conversion ===")

# Step 1: Load model
print("[1/4] Loading SenseVoiceSmall...")
t0 = time.perf_counter()
from funasr import AutoModel
import torch

model = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True)
pt_model = model.model
pt_model.eval()
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

# Step 2: Extract CTC linear head
print("[2/4] Extracting CTC head (ctc.ctc_lo: Linear 512->25055)...")
ctc_lo = pt_model.ctc.ctc_lo
print(f"  Layer: {ctc_lo.__class__.__name__}")
print(f"  Weight shape: {ctc_lo.weight.shape}")
print(f"  Bias shape: {ctc_lo.bias.shape if ctc_lo.bias is not None else 'None'}")


# Create a standalone Linear module for clean conversion
class CTCHeadWrapper(torch.nn.Module):
    """Wraps CTC linear head for OpenVINO conversion."""

    def __init__(self, ctc_linear):
        super().__init__()
        self.ctc_lo = ctc_linear

    def forward(self, x):
        # x: (batch, frames, 512) -> (batch, frames, 25055)
        return self.ctc_lo(x)


ctc_wrapper = CTCHeadWrapper(ctc_lo)
ctc_wrapper.eval()

# Test forward pass
batch_size = 1
num_frames = 100
hidden_dim = 512
test_input = torch.randn(batch_size, num_frames, hidden_dim)
with torch.no_grad():
    test_output = ctc_wrapper(test_input)
print(f"  Forward test: input {test_input.shape} -> output {test_output.shape}")

# Step 3: Convert to OpenVINO IR
print("[3/4] Converting to OpenVINO IR...")
import openvino
from openvino import convert_model

t1 = time.perf_counter()
try:
    ov_model = convert_model(
        ctc_wrapper,
        example_input=test_input,
        share_weights=False,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_path = os.path.join(OUTPUT_DIR, "ctc_head.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=True)
    print(f"  Converted in {time.perf_counter() - t1:.1f}s")
    print(f"  Saved: {xml_path}")
except Exception as e:
    print(f"  convert_model failed: {e}")
    print("  Trying torch.jit.trace approach...")
    with torch.no_grad():
        traced = torch.jit.trace(ctc_wrapper, test_input, strict=False)
    ov_model = convert_model(traced)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_path = os.path.join(OUTPUT_DIR, "ctc_head.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=True)
    print(f"  Traced + converted in {time.perf_counter() - t1:.1f}s")
    print(f"  Saved: {xml_path}")

# Step 4: Verify on NPU/GPU/CPU
print("[4/4] Verifying IR and testing devices...")
if os.path.exists(xml_path):
    for f in os.listdir(OUTPUT_DIR):
        if "ctc" in f:
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
            print(f"  {f}: {sz:.1f} MB")

    core = openvino.Core()
    devices = core.available_devices
    print(f"  Available devices: {devices}")

    model_ir = core.read_model(xml_path)
    print(
        f"  IR inputs: {[(i.get_node().get_friendly_name(), i.get_shape()) for i in model_ir.inputs]}"
    )
    print(
        f"  IR outputs: {[(o.get_node().get_friendly_name(), o.get_shape()) for o in model_ir.outputs]}"
    )

    for dev in ["NPU", "GPU", "CPU"]:
        try:
            compiled = core.compile_model(model_ir, dev)
            input_data = np.random.randn(1, num_frames, hidden_dim).astype(np.float32)
            result = compiled({0: input_data})
            output_keys = list(result.keys())
            output_shapes = {k: result[k].shape for k in output_keys}
            print(f"  {dev} inference OK: outputs = {output_shapes}")

            # Compare with PyTorch reference
            with torch.no_grad():
                ref_output = ctc_wrapper(torch.from_numpy(input_data)).numpy()
            ov_output = result[output_keys[0]]
            max_diff = np.max(np.abs(ref_output - ov_output))
            print(f"    Max diff vs PyTorch: {max_diff:.6f}")
            break
        except Exception as e:
            print(f"  {dev} failed: {e}")

print("=== Done ===")
